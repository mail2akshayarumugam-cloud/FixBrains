# ══════════════════════════════════════════════════════════════
# Train 3D U-Net — Pure R (torch)
# Uses exported .pt tensors from data/training/
# ══════════════════════════════════════════════════════════════

setwd("C:/Users/mail2/OneDrive/Desktop/R project 1")

library(torch)
library(tidyverse)

source("R/utils.R")
source("R/03_model_3d_unet.R")

TRAINING_DIR <- "data/training"
MODELS_PATH  <- "models"
ensure_dir(MODELS_PATH)

manifest <- read_csv(file.path(TRAINING_DIR, "manifest.csv"), show_col_types = FALSE)
cat("Total samples in manifest:", nrow(manifest), "\n")
cat("Patients:", n_distinct(manifest$patient_id), "\n")

manifest <- manifest %>%
  mutate(
    mri_full = file.path(TRAINING_DIR, mri_file),
    seg_full = file.path(TRAINING_DIR, seg_file)
  )

# ---- Train/Val Split (80/20 by patient) ----

set.seed(42)
patients <- unique(manifest$patient_id)
n_train <- floor(length(patients) * 0.8)
train_patients <- sample(patients, n_train)
val_patients   <- setdiff(patients, train_patients)

train_manifest <- manifest %>% filter(patient_id %in% train_patients)
val_manifest   <- manifest %>% filter(patient_id %in% val_patients)

cat("Train:", nrow(train_manifest), "samples from", length(train_patients), "patients\n")
cat("Val:  ", nrow(val_manifest), "samples from", length(val_patients), "patients\n")

# ---- Train ----

device <- if (cuda_is_available()) "cuda" else "cpu"
cat("Device:", device, "\n\n")

result <- train_unet(
  train_mri_paths = train_manifest$mri_full,
  train_seg_paths = train_manifest$seg_full,
  val_mri_paths   = val_manifest$mri_full,
  val_seg_paths   = val_manifest$seg_full,
  epochs       = 50L,
  batch_size   = 2L,
  lr           = 1e-4,
  in_channels  = 4L,
  n_classes    = 4L,
  base_filters = 16L,
  save_path    = file.path(MODELS_PATH, "unet_3d_best.pt")
)

# ---- Save Training History ----

history_df <- tibble(
  epoch      = seq_along(result$history$train_loss),
  train_loss = result$history$train_loss,
  val_loss   = result$history$val_loss
)
write_csv(history_df, file.path(MODELS_PATH, "unet_training_history.csv"))

# ---- Plot Training Curves ----

library(ggplot2)
library(patchwork)

p1 <- history_df %>%
  pivot_longer(c(train_loss, val_loss), names_to = "Split", values_to = "Loss") %>%
  mutate(Split = recode(Split, train_loss = "Train", val_loss = "Val")) %>%
  ggplot(aes(epoch, Loss, color = Split)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 1.5, alpha = 0.5) +
  labs(title = "3D U-Net Training Loss", x = "Epoch", y = "Combined Loss (Dice + CE)") +
  scale_color_manual(values = c(Train = "#2563eb", Val = "#dc2626")) +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

print(p1)
ggsave(file.path("results/figures", "13_unet_training_curves.png"),
       p1, width = 10, height = 5, dpi = 150)

cat("\n══════════════════════════════════════════\n")
cat("  Training Complete\n")
cat("══════════════════════════════════════════\n")
cat("  Best model saved: models/unet_3d_best.pt\n")
cat("  History saved:    models/unet_training_history.csv\n")
cat("  Figure saved:     results/figures/13_unet_training_curves.png\n")
cat("══════════════════════════════════════════\n")
