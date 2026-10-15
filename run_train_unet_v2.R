# ══════════════════════════════════════════════════════════════
# Train 3D U-Net v2 — Improved (Pure R torch, CPU-friendly)
#
# Improvements over v1:
#   - Data augmentation (flips, intensity jitter)
#   - Class-weighted Dice loss (boosts necrosis)
#   - Dice score tracking per class
#   - Learning rate scheduling (cosine)
#   - Early stopping with patience
#   - Checkpointing (resume from crash)
#   - Gradient clipping
#
# Run: Rscript run_train_unet_v2.R
# CPU: ~25-35 min/epoch, ~2-3 days for 100 epochs
# GPU: ~1-2 min/epoch, ~2-3 hours for 100 epochs
# ══════════════════════════════════════════════════════════════

setwd("C:/Users/mail2/OneDrive/Desktop/R project 1")

library(torch)
library(tidyverse)

source("R/utils.R")
source("R/03_model_3d_unet.R")

TRAINING_DIR <- "data/training"
MODELS_PATH  <- "models"
CHECKPOINT_DIR <- "data/processed/checkpoints"
ensure_dir(MODELS_PATH)
ensure_dir(CHECKPOINT_DIR)

# ── Config ───────────────────────────────────────────────────

EPOCHS       <- 100L
BATCH_SIZE   <- 1L       # 1 for CPU (saves memory)
LR_INIT      <- 3e-4
LR_MIN       <- 1e-6
BASE_FILTERS <- 16L
N_CLASSES    <- 4L
IN_CHANNELS  <- 4L
PATIENCE     <- 20L
CLASS_WEIGHTS <- c(0.5, 3.0, 1.5, 2.0)  # bg, necrosis(boost), edema, enhancing

cat("══════════════════════════════════════════\n")
cat("  3D U-Net Training v2 (Improved)\n")
cat("══════════════════════════════════════════\n\n")

# ── Load manifest ────────────────────────────────────────────

manifest <- read_csv(file.path(TRAINING_DIR, "manifest.csv"), show_col_types = FALSE)
cat("Total samples:", nrow(manifest), "\n")
cat("Patients:", n_distinct(manifest$patient_id), "\n")

manifest <- manifest %>%
  mutate(
    mri_full = file.path(TRAINING_DIR, mri_file),
    seg_full = file.path(TRAINING_DIR, seg_file)
  ) %>%
  filter(file.exists(mri_full), file.exists(seg_full))

cat("Valid samples:", nrow(manifest), "\n\n")

# ── Train/Val Split ──────────────────────────────────────────

set.seed(42)
patients <- unique(manifest$patient_id)
n_train <- floor(length(patients) * 0.8)
train_patients <- sample(patients, n_train)
val_patients <- setdiff(patients, train_patients)

train_manifest <- manifest %>% filter(patient_id %in% train_patients)
val_manifest <- manifest %>% filter(patient_id %in% val_patients)

cat("Train:", nrow(train_manifest), "samples (", length(train_patients), "patients)\n")
cat("Val:  ", nrow(val_manifest), "samples (", length(val_patients), "patients)\n\n")

# ── Data Augmentation ────────────────────────────────────────

augment_3d <- function(mri_tensor, seg_tensor) {
  # Random horizontal flip
  if (runif(1) > 0.5) {
    mri_tensor <- torch_flip(mri_tensor, 2)  # flip x
    seg_tensor <- torch_flip(seg_tensor, 1)
  }
  # Random vertical flip
  if (runif(1) > 0.5) {
    mri_tensor <- torch_flip(mri_tensor, 3)  # flip y
    seg_tensor <- torch_flip(seg_tensor, 2)
  }
  # Random depth flip
  if (runif(1) > 0.5) {
    mri_tensor <- torch_flip(mri_tensor, 4)  # flip z
    seg_tensor <- torch_flip(seg_tensor, 3)
  }
  # Random intensity jitter (MRI only, not seg)
  if (runif(1) > 0.5) {
    scale <- 0.9 + runif(1) * 0.2  # 0.9 to 1.1
    shift <- (runif(1) - 0.5) * 0.2  # -0.1 to 0.1
    mri_tensor <- mri_tensor * scale + shift
  }
  # Random Gaussian noise
  if (runif(1) > 0.7) {
    noise <- torch_randn_like(mri_tensor) * 0.05
    mri_tensor <- mri_tensor + noise
  }
  list(mri = mri_tensor, seg = seg_tensor)
}

# ── Dataset with augmentation ────────────────────────────────

aug_dataset <- dataset(
  "AugTumorDataset",
  initialize = function(mri_paths, seg_paths, augment = FALSE) {
    self$mri_paths <- mri_paths
    self$seg_paths <- seg_paths
    self$augment <- augment
  },
  .getitem = function(i) {
    mri <- torch_load(self$mri_paths[i])$to(dtype = torch_float())
    seg <- torch_load(self$seg_paths[i])$to(dtype = torch_long())

    if (mri$dim() == 3) mri <- mri$unsqueeze(1)
    if (mri$dim() == 4 && mri$size(4) <= 4) mri <- mri$permute(c(4, 1, 2, 3))

    if (self$augment) {
      aug <- augment_3d(mri, seg)
      mri <- aug$mri
      seg <- aug$seg
    }

    list(x = mri, y = seg)
  },
  .length = function() length(self$mri_paths)
)

# ── Weighted Dice Loss ───────────────────────────────────────

weighted_dice_loss <- function(pred, target, class_weights, smooth = 1.0) {
  pred_soft <- nnf_softmax(pred, dim = 2)
  n_classes <- pred$size(2)
  total_loss <- torch_tensor(0, device = pred$device)

  for (c in seq_len(n_classes)) {
    pred_c <- pred_soft[, c, , , ]
    target_c <- (target == (c - 1L))$to(dtype = torch_float())

    intersection <- (pred_c * target_c)$sum()
    union_sum <- pred_c$sum() + target_c$sum()
    dice <- (2 * intersection + smooth) / (union_sum + smooth)

    total_loss <- total_loss + class_weights[c] * (1 - dice)
  }

  total_loss / sum(class_weights)
}

improved_loss <- function(pred, target, class_weights) {
  w_tensor <- torch_tensor(class_weights, dtype = torch_float(), device = pred$device)
  ce <- nnf_cross_entropy(pred, target$to(dtype = torch_long()), weight = w_tensor)
  dl <- weighted_dice_loss(pred, target, class_weights)
  0.4 * ce + 0.6 * dl
}

# ── Dice Score Calculator ────────────────────────────────────

compute_dice_scores <- function(pred, target, n_classes = 4L) {
  pred_class <- pred$argmax(dim = 2)$squeeze(1)
  target <- target$squeeze(1)
  dices <- numeric(n_classes)
  for (c in seq_len(n_classes)) {
    p <- (pred_class == (c - 1L))
    t <- (target == (c - 1L))
    intersection <- (p & t)$sum()$item()
    total <- p$sum()$item() + t$sum()$item()
    dices[c] <- if (total > 0) 2 * intersection / total else NA_real_
  }
  names(dices) <- c("bg", "necrosis", "edema", "enhancing")
  dices
}

# ── Cosine LR Schedule ───────────────────────────────────────

cosine_lr <- function(epoch, max_epochs, lr_init, lr_min) {
  lr_min + 0.5 * (lr_init - lr_min) * (1 + cos(pi * epoch / max_epochs))
}

# ── Training Loop ────────────────────────────────────────────

device <- if (cuda_is_available()) "cuda" else "cpu"
cat("Device:", device, "\n")

train_ds <- aug_dataset(train_manifest$mri_full, train_manifest$seg_full, augment = TRUE)
val_ds <- aug_dataset(val_manifest$mri_full, val_manifest$seg_full, augment = FALSE)
train_dl <- dataloader(train_ds, batch_size = BATCH_SIZE, shuffle = TRUE)
val_dl <- dataloader(val_ds, batch_size = BATCH_SIZE, shuffle = FALSE)

# Check for checkpoint to resume from
checkpoint_path <- file.path(CHECKPOINT_DIR, "unet_v2_checkpoint.pt")
history_path <- file.path(CHECKPOINT_DIR, "unet_v2_history.csv")

model <- unet_3d(IN_CHANNELS, N_CLASSES, BASE_FILTERS)
model <- model$to(device = device)

n_params <- sum(sapply(model$parameters, function(p) prod(p$shape)))
cat("Parameters:", format(n_params, big.mark = ","), "\n")

optimizer <- optim_adam(model$parameters, lr = LR_INIT, weight_decay = 1e-5)

start_epoch <- 1L
best_val_dice <- 0
patience_counter <- 0
history <- tibble(
  epoch = integer(), train_loss = numeric(), val_loss = numeric(),
  val_dice = numeric(), val_dice_nec = numeric(),
  val_dice_edema = numeric(), val_dice_enh = numeric(), lr = numeric()
)

if (file.exists(history_path)) {
  history <- read_csv(history_path, show_col_types = FALSE)
  start_epoch <- nrow(history) + 1L
  best_val_dice <- max(history$val_dice, na.rm = TRUE)
  cat("Resuming from epoch", start_epoch, "| Best Dice so far:", round(best_val_dice, 4), "\n")
}
if (file.exists(checkpoint_path)) {
  tryCatch({
    saved <- torch_load(checkpoint_path)
    model$load_state_dict(saved)
    cat("Loaded checkpoint weights\n")
  }, error = function(e) cat("Could not load checkpoint:", e$message, "\n"))
}

cat("\nStarting training...\n")
cat(strrep("-", 90), "\n")
cat(sprintf("%-6s | %-11s | %-11s | %-10s | %-8s | %-8s | %-8s | %-8s\n",
            "Epoch", "Train Loss", "Val Loss", "Val Dice", "Necrosis", "Edema", "Enhanc.", "LR"))
cat(strrep("-", 90), "\n")

save_path <- file.path(MODELS_PATH, "unet_3d_v2_best.pt")

for (epoch in seq(start_epoch, EPOCHS)) {
  # Update learning rate
  current_lr <- cosine_lr(epoch, EPOCHS, LR_INIT, LR_MIN)
  for (pg in optimizer$param_groups) pg$lr <- current_lr

  # ── Train ──
  model$train()
  train_losses <- c()
  batch_count <- 0L

  coro::loop(for (batch in train_dl) {
    x <- batch$x$to(device = device)
    y <- batch$y$to(device = device)

    optimizer$zero_grad()
    pred <- model(x)
    loss <- improved_loss(pred, y, CLASS_WEIGHTS)
    loss$backward()
    nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
    optimizer$step()

    train_losses <- c(train_losses, loss$item())
    batch_count <- batch_count + 1L

    if (batch_count %% 50 == 0) {
      cat(sprintf("  [Epoch %d] Batch %d/%d | Loss: %.4f\r",
                  epoch, batch_count, ceiling(nrow(train_manifest)/BATCH_SIZE),
                  mean(tail(train_losses, 10))))
    }
  })

  avg_train_loss <- mean(train_losses)

  # ── Validate ──
  model$eval()
  val_losses <- c()
  all_dices <- list()

  with_no_grad({
    coro::loop(for (batch in val_dl) {
      x <- batch$x$to(device = device)
      y <- batch$y$to(device = device)

      pred <- model(x)
      loss <- improved_loss(pred, y, CLASS_WEIGHTS)
      val_losses <- c(val_losses, loss$item())

      dices <- compute_dice_scores(pred, y, N_CLASSES)
      all_dices <- c(all_dices, list(dices))
    })
  })

  avg_val_loss <- mean(val_losses)
  dice_mat <- do.call(rbind, all_dices)
  avg_dice_nec <- mean(dice_mat[, "necrosis"], na.rm = TRUE)
  avg_dice_edema <- mean(dice_mat[, "edema"], na.rm = TRUE)
  avg_dice_enh <- mean(dice_mat[, "enhancing"], na.rm = TRUE)
  avg_val_dice <- mean(c(avg_dice_nec, avg_dice_edema, avg_dice_enh), na.rm = TRUE)

  # Log
  is_best <- ""
  if (avg_val_dice > best_val_dice) {
    best_val_dice <- avg_val_dice
    torch_save(model$state_dict(), save_path)
    patience_counter <- 0
    is_best <- " *BEST*"
  } else {
    patience_counter <- patience_counter + 1
  }

  # Save checkpoint every epoch
  torch_save(model$state_dict(), checkpoint_path)

  history <- bind_rows(history, tibble(
    epoch = epoch, train_loss = avg_train_loss, val_loss = avg_val_loss,
    val_dice = avg_val_dice, val_dice_nec = avg_dice_nec,
    val_dice_edema = avg_dice_edema, val_dice_enh = avg_dice_enh, lr = current_lr
  ))
  write_csv(history, history_path)

  cat(sprintf("%-6d | %-11.4f | %-11.4f | %-10.4f | %-8.4f | %-8.4f | %-8.4f | %-8.6f%s\n",
              epoch, avg_train_loss, avg_val_loss, avg_val_dice,
              avg_dice_nec, avg_dice_edema, avg_dice_enh, current_lr, is_best))

  # Early stopping
  if (patience_counter >= PATIENCE) {
    cat("\nEarly stopping at epoch", epoch, "(no improvement for", PATIENCE, "epochs)\n")
    break
  }

  gc()
}

# ── Final Results ────────────────────────────────────────────

cat("\n")
cat(strrep("=", 60), "\n")
cat("  Training Complete\n")
cat(strrep("=", 60), "\n")
cat("  Best Val Dice:     ", round(best_val_dice, 4), "\n")
cat("  Best Dice Necrosis:", round(max(history$val_dice_nec, na.rm = TRUE), 4), "\n")
cat("  Best Dice Edema:   ", round(max(history$val_dice_edema, na.rm = TRUE), 4), "\n")
cat("  Best Dice Enhancing:", round(max(history$val_dice_enh, na.rm = TRUE), 4), "\n")
cat("  Epochs trained:    ", nrow(history), "\n")
cat("  Model saved:       ", save_path, "\n")
cat("  History saved:     ", history_path, "\n")
cat(strrep("=", 60), "\n")

# ── Training Curves ──────────────────────────────────────────

library(ggplot2)
library(patchwork)

p_loss <- history %>%
  pivot_longer(c(train_loss, val_loss), names_to = "Split", values_to = "Loss") %>%
  mutate(Split = recode(Split, train_loss = "Train", val_loss = "Val")) %>%
  ggplot(aes(epoch, Loss, color = Split)) +
  geom_line(linewidth = 1.2) +
  scale_color_manual(values = c(Train = "#2563eb", Val = "#dc2626")) +
  labs(title = "Training & Validation Loss") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

p_dice <- history %>%
  select(epoch, Necrosis = val_dice_nec, Edema = val_dice_edema, Enhancing = val_dice_enh) %>%
  pivot_longer(-epoch, names_to = "Region", values_to = "Dice") %>%
  ggplot(aes(epoch, Dice, color = Region)) +
  geom_line(linewidth = 1.2) +
  scale_color_manual(values = c(Necrosis = "#ef4444", Edema = "#f59e0b", Enhancing = "#22c55e")) +
  labs(title = "Per-Class Dice Score") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

p_lr <- history %>%
  ggplot(aes(epoch, lr)) +
  geom_line(linewidth = 1, color = "#a855f7") +
  labs(title = "Learning Rate (Cosine)") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

p_all <- (p_loss | p_dice) / p_lr + plot_layout(heights = c(2, 1)) +
  plot_annotation(
    title = "3D U-Net v2 Training — Improved",
    subtitle = paste("Best Val Dice:", round(best_val_dice, 4),
                     "| Epochs:", nrow(history),
                     "| Augmentation: flips + intensity jitter + noise"),
    theme = theme(plot.title = element_text(face = "bold", size = 16))
  )
print(p_all)
ggsave(file.path("results/figures", "13_unet_v2_training.png"),
       p_all, width = 14, height = 8, dpi = 150)

cat("\nDone. Training curves saved to results/figures/13_unet_v2_training.png\n")
