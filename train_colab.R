# ══════════════════════════════════════════════════════════════
# 3D U-Net v2 Training — Pure R (Google Colab T4 GPU)
# ══════════════════════════════════════════════════════════════
#
# HOW TO RUN IN GOOGLE COLAB:
#
#   1. Go to colab.research.google.com/#create=true&language=r
#      (This opens Colab with R runtime)
#
#   2. If that doesn't work:
#      - Open any notebook
#      - Runtime → Change runtime type → R → T4 GPU → Save
#
#   3. Upload data/training/ folder to Google Drive:
#      Google Drive / R_project / training /
#        manifest.csv
#        mri/   (599 .pt files)
#        seg/   (599 .pt files)
#
#   4. Copy this entire file into ONE cell and run it
#      Or split at the ── CELL markers into separate cells
#
#   5. When done, download from Drive:
#      R_project/models/unet_3d_v2_jit.pt → models/unet_3d_trained_jit.pt
#
# Expected time: ~1-2 hours on T4 GPU
# ══════════════════════════════════════════════════════════════

# ── CELL 1: Setup & Install ─────────────────────────────────

options(timeout = 600)
install.packages("torch", repos = "https://cloud.r-project.org")
Sys.setenv(TORCH_INSTALL = "1", CUDA = "")
library(torch)
if (!torch_is_installed()) {
  install_torch(type = "cpu", reinstall = TRUE)
}
cat("Torch installed:", torch_is_installed(), "\n")
cat("Test:", as.numeric(torch_tensor(1) + torch_tensor(1)), "\n")

device <- "cpu"
cat("Training on:", device, "\n")
cat("Note: CPU on Colab (~4-6 hrs for 150 epochs)\n\n")

install.packages("coro", repos = "https://cloud.r-project.org")
install.packages("readr", repos = "https://cloud.r-project.org")

# Mount Google Drive (Colab auto-mounts, but just in case)
if (!dir.exists("/content/drive")) {
  system("apt-get install -y -qq fuse 2>/dev/null", intern = TRUE)
}

DRIVE_ROOT <- "/content/drive/MyDrive/R_project"
TRAINING_DIR <- file.path(DRIVE_ROOT, "data", "training")
MODEL_DIR <- file.path(DRIVE_ROOT, "models")
dir.create(MODEL_DIR, showWarnings = FALSE, recursive = TRUE)

if (!file.exists(file.path(TRAINING_DIR, "manifest.csv"))) {
  stop("Training data not found! Upload data/training/ to Google Drive under R_project/")
}

cat("Training data found at:", TRAINING_DIR, "\n")

# ── CELL 2: 3D U-Net Architecture ───────────────────────────

conv_block_3d <- nn_module(
  "ConvBlock3D",
  initialize = function(in_ch, out_ch) {
    self$conv1 <- nn_conv3d(in_ch, out_ch, kernel_size = 3, padding = 1)
    self$bn1 <- nn_batch_norm3d(out_ch)
    self$conv2 <- nn_conv3d(out_ch, out_ch, kernel_size = 3, padding = 1)
    self$bn2 <- nn_batch_norm3d(out_ch)
    self$relu <- nn_relu()
  },
  forward = function(x) {
    x <- self$relu(self$bn1(self$conv1(x)))
    x <- self$relu(self$bn2(self$conv2(x)))
    x
  }
)

encoder_block_3d <- nn_module(
  "EncoderBlock3D",
  initialize = function(in_ch, out_ch) {
    self$conv_block <- conv_block_3d(in_ch, out_ch)
    self$pool <- nn_max_pool3d(kernel_size = 2, stride = 2)
  },
  forward = function(x) {
    feat <- self$conv_block(x)
    pooled <- self$pool(feat)
    list(features = feat, pooled = pooled)
  }
)

decoder_block_3d <- nn_module(
  "DecoderBlock3D",
  initialize = function(in_ch, out_ch) {
    self$upconv <- nn_conv_transpose3d(in_ch, out_ch, kernel_size = 2, stride = 2)
    self$conv_block <- conv_block_3d(out_ch * 2L, out_ch)
  },
  forward = function(x, skip) {
    x <- self$upconv(x)
    diff_d <- skip$size(3) - x$size(3)
    diff_h <- skip$size(4) - x$size(4)
    diff_w <- skip$size(5) - x$size(5)
    x <- nnf_pad(x, c(diff_w %/% 2, diff_w - diff_w %/% 2,
                       diff_h %/% 2, diff_h - diff_h %/% 2,
                       diff_d %/% 2, diff_d - diff_d %/% 2))
    x <- torch_cat(list(x, skip), dim = 2)
    self$conv_block(x)
  }
)

unet_3d <- nn_module(
  "UNet3D",
  initialize = function(in_ch = 4L, n_classes = 4L, base = 32L) {
    self$enc1 <- encoder_block_3d(in_ch, base)
    self$enc2 <- encoder_block_3d(base, base * 2L)
    self$enc3 <- encoder_block_3d(base * 2L, base * 4L)
    self$enc4 <- encoder_block_3d(base * 4L, base * 8L)
    self$bottleneck <- conv_block_3d(base * 8L, base * 16L)
    self$dec4 <- decoder_block_3d(base * 16L, base * 8L)
    self$dec3 <- decoder_block_3d(base * 8L, base * 4L)
    self$dec2 <- decoder_block_3d(base * 4L, base * 2L)
    self$dec1 <- decoder_block_3d(base * 2L, base)
    self$final_conv <- nn_conv3d(base, n_classes, kernel_size = 1)
  },
  forward = function(x) {
    e1 <- self$enc1(x); e2 <- self$enc2(e1$pooled)
    e3 <- self$enc3(e2$pooled); e4 <- self$enc4(e3$pooled)
    b <- self$bottleneck(e4$pooled)
    d4 <- self$dec4(b, e4$features); d3 <- self$dec3(d4, e3$features)
    d2 <- self$dec2(d3, e2$features); d1 <- self$dec1(d2, e1$features)
    self$final_conv(d1)
  }
)

# ── CELL 3: Data Augmentation ────────────────────────────────

augment_3d <- function(mri_tensor, seg_tensor) {
  if (runif(1) > 0.5) { mri_tensor <- torch_flip(mri_tensor, 2); seg_tensor <- torch_flip(seg_tensor, 1) }
  if (runif(1) > 0.5) { mri_tensor <- torch_flip(mri_tensor, 3); seg_tensor <- torch_flip(seg_tensor, 2) }
  if (runif(1) > 0.5) { mri_tensor <- torch_flip(mri_tensor, 4); seg_tensor <- torch_flip(seg_tensor, 3) }
  if (runif(1) > 0.5) {
    scale <- 0.9 + runif(1) * 0.2
    shift <- (runif(1) - 0.5) * 0.2
    mri_tensor <- mri_tensor * scale + shift
  }
  if (runif(1) > 0.7) {
    mri_tensor <- mri_tensor + torch_randn_like(mri_tensor) * 0.05
  }
  list(mri = mri_tensor, seg = seg_tensor)
}

# ── CELL 4: Dataset ──────────────────────────────────────────

tumor_dataset <- dataset(
  "TumorDataset",
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
      mri <- aug$mri; seg <- aug$seg
    }
    list(x = mri, y = seg)
  },
  .length = function() length(self$mri_paths)
)

# ── CELL 5: Loss Functions ───────────────────────────────────

CLASS_WEIGHTS <- c(0.5, 3.0, 1.5, 2.0)

weighted_dice_loss <- function(pred, target, weights = CLASS_WEIGHTS, smooth = 1.0) {
  pred_soft <- nnf_softmax(pred, dim = 2)
  n_classes <- pred$size(2)
  total <- torch_tensor(0, device = pred$device)
  for (c in seq_len(n_classes)) {
    pred_c <- pred_soft[, c, , , ]
    target_c <- (target == (c - 1L))$to(dtype = torch_float())
    inter <- (pred_c * target_c)$sum()
    denom <- pred_c$sum() + target_c$sum()
    dice <- (2 * inter + smooth) / (denom + smooth)
    total <- total + weights[c] * (1 - dice)
  }
  total / sum(weights)
}

combined_loss <- function(pred, target) {
  w <- torch_tensor(CLASS_WEIGHTS, dtype = torch_float(), device = pred$device)
  ce <- nnf_cross_entropy(pred, target$to(dtype = torch_long()), weight = w)
  dl <- weighted_dice_loss(pred, target)
  0.4 * ce + 0.6 * dl
}

compute_dice_per_class <- function(pred, target, n_classes = 4L) {
  pred_cls <- pred$argmax(dim = 2)$squeeze(1)
  target <- target$squeeze(1)
  dices <- numeric(n_classes)
  names(dices) <- c("bg", "necrosis", "edema", "enhancing")
  for (c in seq_len(n_classes)) {
    p <- (pred_cls == (c - 1L))
    t <- (target == (c - 1L))
    inter <- (p & t)$sum()$item()
    total <- p$sum()$item() + t$sum()$item()
    dices[c] <- if (total > 0) 2 * inter / total else NA_real_
  }
  dices
}

# ── CELL 6: Load Data & Split ────────────────────────────────

library(readr)
manifest <- read_csv(file.path(TRAINING_DIR, "manifest.csv"), show_col_types = FALSE)
manifest <- manifest[manifest$tumor_voxels > 0, ]
cat("Samples with tumor:", nrow(manifest), "\n")
cat("Patients:", length(unique(manifest$patient_id)), "\n")

manifest$mri_full <- file.path(TRAINING_DIR, manifest$mri_file)
manifest$seg_full <- file.path(TRAINING_DIR, manifest$seg_file)

set.seed(42)
patients <- unique(manifest$patient_id)
n_train <- floor(length(patients) * 0.8)
train_patients <- sample(patients, n_train)
val_patients <- setdiff(patients, train_patients)

train_df <- manifest[manifest$patient_id %in% train_patients, ]
val_df <- manifest[manifest$patient_id %in% val_patients, ]

cat("Train:", nrow(train_df), "samples (", length(train_patients), "patients)\n")
cat("Val:  ", nrow(val_df), "samples (", length(val_patients), "patients)\n\n")

train_ds <- tumor_dataset(train_df$mri_full, train_df$seg_full, augment = TRUE)
val_ds <- tumor_dataset(val_df$mri_full, val_df$seg_full, augment = FALSE)
train_dl <- dataloader(train_ds, batch_size = 4L, shuffle = TRUE)
val_dl <- dataloader(val_ds, batch_size = 4L, shuffle = FALSE)

# ── CELL 7: Train ────────────────────────────────────────────

BASE_FILTERS <- 32L
EPOCHS <- 150L
LR_INIT <- 3e-4
LR_MIN <- 1e-6
PATIENCE <- 20L

model <- unet_3d(in_ch = 4L, n_classes = 4L, base = BASE_FILTERS)
model <- model$to(device = device)

n_params <- sum(sapply(model$parameters, function(p) prod(p$shape)))
cat("Parameters:", format(n_params, big.mark = ","), "\n")
cat("Base filters:", BASE_FILTERS, "\n")
cat("Device:", device, "\n\n")

optimizer <- optim_adam(model$parameters, lr = LR_INIT, weight_decay = 1e-5)

cosine_lr <- function(epoch, max_ep, lr_max, lr_min) {
  lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * epoch / max_ep))
}

best_dice <- 0
patience_counter <- 0
history <- data.frame(
  epoch = integer(), train_loss = numeric(), val_loss = numeric(),
  val_dice = numeric(), dice_nec = numeric(), dice_edema = numeric(),
  dice_enh = numeric(), lr = numeric()
)

save_path <- file.path(MODEL_DIR, "unet_3d_v2_best.pt")
checkpoint_path <- file.path(MODEL_DIR, "unet_3d_v2_checkpoint.pt")
history_path <- file.path(MODEL_DIR, "unet_v2_history.csv")

# Resume from checkpoint if exists
if (file.exists(history_path)) {
  history <- read_csv(history_path, show_col_types = FALSE)
  best_dice <- max(history$val_dice, na.rm = TRUE)
  cat("Resuming from epoch", nrow(history) + 1, "| Best dice:", round(best_dice, 4), "\n")
}
if (file.exists(checkpoint_path)) {
  tryCatch({
    model$load_state_dict(torch_load(checkpoint_path))
    cat("Loaded checkpoint\n")
  }, error = function(e) cat("Could not load checkpoint\n"))
}

start_epoch <- nrow(history) + 1L

cat(strrep("=", 95), "\n")
cat(sprintf("%-6s | %-11s | %-11s | %-10s | %-8s | %-8s | %-8s | %-10s\n",
            "Epoch", "Train Loss", "Val Loss", "Val Dice", "Necrosis", "Edema", "Enhanc.", "LR"))
cat(strrep("=", 95), "\n")

t_start <- Sys.time()

for (epoch in seq(start_epoch, EPOCHS)) {
  current_lr <- cosine_lr(epoch, EPOCHS, LR_INIT, LR_MIN)
  for (pg in optimizer$param_groups) pg$lr <- current_lr

  # Train
  model$train()
  train_losses <- c()
  coro::loop(for (batch in train_dl) {
    x <- batch$x$to(device = device)
    y <- batch$y$to(device = device)
    optimizer$zero_grad()
    pred <- model(x)
    loss <- combined_loss(pred, y)
    loss$backward()
    nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
    optimizer$step()
    train_losses <- c(train_losses, loss$item())
  })
  avg_train <- mean(train_losses)

  # Validate
  model$eval()
  val_losses <- c()
  all_dices <- list()
  with_no_grad({
    coro::loop(for (batch in val_dl) {
      x <- batch$x$to(device = device)
      y <- batch$y$to(device = device)
      pred <- model(x)
      loss <- combined_loss(pred, y)
      val_losses <- c(val_losses, loss$item())
      all_dices <- c(all_dices, list(compute_dice_per_class(pred, y)))
    })
  })

  avg_val <- mean(val_losses)
  dice_mat <- do.call(rbind, all_dices)
  d_nec <- mean(dice_mat[, "necrosis"], na.rm = TRUE)
  d_edema <- mean(dice_mat[, "edema"], na.rm = TRUE)
  d_enh <- mean(dice_mat[, "enhancing"], na.rm = TRUE)
  d_mean <- mean(c(d_nec, d_edema, d_enh), na.rm = TRUE)

  marker <- ""
  if (d_mean > best_dice) {
    best_dice <- d_mean
    torch_save(model$state_dict(), save_path)
    patience_counter <- 0
    marker <- " *BEST*"
  } else {
    patience_counter <- patience_counter + 1
  }

  torch_save(model$state_dict(), checkpoint_path)

  history <- rbind(history, data.frame(
    epoch = epoch, train_loss = avg_train, val_loss = avg_val,
    val_dice = d_mean, dice_nec = d_nec, dice_edema = d_edema,
    dice_enh = d_enh, lr = current_lr
  ))
  write_csv(history, history_path)

  elapsed <- as.numeric(difftime(Sys.time(), t_start, units = "mins"))
  eta <- elapsed / (epoch - start_epoch + 1) * (EPOCHS - epoch)

  cat(sprintf("%-6d | %-11.4f | %-11.4f | %-10.4f | %-8.4f | %-8.4f | %-8.4f | %-10.6f%s  [%.0fm / ~%.0fm left]\n",
              epoch, avg_train, avg_val, d_mean, d_nec, d_edema, d_enh, current_lr, marker, elapsed, eta))

  if (patience_counter >= PATIENCE) {
    cat("\nEarly stopping at epoch", epoch, "\n")
    break
  }

  if (epoch %% 10 == 0) { gc(); if (cuda_is_available()) cuda_empty_cache() }
}

# ── CELL 8: Save JIT Model for R ────────────────────────────

cat("\nSaving JIT model for R inference...\n")

model$load_state_dict(torch_load(save_path))
model$eval()
model_cpu <- model$to(device = "cpu")

dummy <- torch_randn(1, 4, 48, 48, 48)
with_no_grad({ traced <- jit_trace(model_cpu, dummy) })
jit_save(traced, file.path(MODEL_DIR, "unet_3d_v2_jit.pt"))

cat("JIT model saved:", file.path(MODEL_DIR, "unet_3d_v2_jit.pt"), "\n")

# Save results
results <- data.frame(
  metric = c("Best Val Dice", "Dice - Necrosis", "Dice - Edema", "Dice - Enhancing",
             "Epochs Trained", "Parameters", "Base Filters"),
  value = c(round(best_dice, 4),
            round(max(history$dice_nec, na.rm = TRUE), 4),
            round(max(history$dice_edema, na.rm = TRUE), 4),
            round(max(history$dice_enh, na.rm = TRUE), 4),
            nrow(history), format(n_params, big.mark = ","), BASE_FILTERS)
)
write_csv(results, file.path(MODEL_DIR, "unet_v2_results.csv"))

cat("\n", strrep("=", 60), "\n")
cat("  TRAINING COMPLETE\n")
cat(strrep("=", 60), "\n")
cat("  Best Val Dice:      ", round(best_dice, 4), "\n")
cat("  Best Necrosis:      ", round(max(history$dice_nec, na.rm = TRUE), 4), "\n")
cat("  Best Edema:         ", round(max(history$dice_edema, na.rm = TRUE), 4), "\n")
cat("  Best Enhancing:     ", round(max(history$dice_enh, na.rm = TRUE), 4), "\n")
cat("  Epochs:             ", nrow(history), "\n")
cat("  Parameters:         ", format(n_params, big.mark = ","), "\n")
cat(strrep("=", 60), "\n")

# ── CELL 9: Training Curves ─────────────────────────────────

if (requireNamespace("ggplot2", quietly = TRUE)) {
  library(ggplot2)

  p1 <- ggplot(history, aes(x = epoch)) +
    geom_line(aes(y = train_loss, color = "Train"), linewidth = 1) +
    geom_line(aes(y = val_loss, color = "Val"), linewidth = 1) +
    scale_color_manual(values = c(Train = "#2563eb", Val = "#dc2626")) +
    labs(title = "Loss", y = "Loss", color = "") +
    theme_minimal(base_size = 12) +
    theme(plot.title = element_text(face = "bold"))

  p2 <- ggplot(history, aes(x = epoch)) +
    geom_line(aes(y = dice_nec, color = "Necrosis"), linewidth = 1) +
    geom_line(aes(y = dice_edema, color = "Edema"), linewidth = 1) +
    geom_line(aes(y = dice_enh, color = "Enhancing"), linewidth = 1) +
    scale_color_manual(values = c(Necrosis = "#ef4444", Edema = "#f59e0b", Enhancing = "#22c55e")) +
    labs(title = "Per-Class Dice", y = "Dice Score", color = "") +
    theme_minimal(base_size = 12) +
    theme(plot.title = element_text(face = "bold"))

  ggsave(file.path(MODEL_DIR, "training_curves_v2.png"), p1, width = 8, height = 4, dpi = 150)
  ggsave(file.path(MODEL_DIR, "dice_curves_v2.png"), p2, width = 8, height = 4, dpi = 150)
  cat("Plots saved to Drive\n")
} else {
  cat("ggplot2 not available — skipping plots\n")
}

cat("\n\nDONE! Download these files from Google Drive:\n")
cat("  R_project/models/unet_3d_v2_jit.pt    → models/unet_3d_trained_jit.pt\n")
cat("  R_project/models/unet_v2_results.csv   → models/unet_results.csv\n")
cat("  R_project/models/training_curves_v2.png → models/training_curves.png\n")
