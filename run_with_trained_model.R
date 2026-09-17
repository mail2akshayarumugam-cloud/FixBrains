# ══════════════════════════════════════════════════════════════
# End-to-End Pipeline with Trained 3D U-Net
# Run this AFTER training the model on Colab
# Place trained model at: models/unet_3d_trained.pt
# ══════════════════════════════════════════════════════════════

setwd("C:/Users/mail2/OneDrive/Desktop/R project 1")

suppressPackageStartupMessages({
  library(tidyverse)
  library(RNifti)
  library(data.table)
  library(ggplot2)
  library(patchwork)
  library(viridis)
  library(torch)
  library(caret)
  library(irr)
  library(pROC)
})

source("R/utils.R")
source("R/01_data_loader.R")
source("R/02_preprocessing.R")
source("R/03_model_3d_unet.R")
source("R/04_region_extractor.R")
source("R/05_region_tracker.R")
source("R/06_feature_extraction.R")
source("R/07_progression_engine.R")
source("R/08_model_lstm.R")
source("R/09_clinical_validation.R")
source("R/11_inference.R")

DATA_PATH <- "data/raw/LUMIERE"
MODEL_PATH <- "models/unet_3d_trained_jit.pt"
RESULTS_PATH <- "results"
FIGURES_PATH <- file.path(RESULTS_PATH, "figures")
TABLES_PATH <- file.path(RESULTS_PATH, "tables")

for (p in c(RESULTS_PATH, FIGURES_PATH, TABLES_PATH)) ensure_dir(p)

# ---- Check trained model ----

if (!file.exists(MODEL_PATH)) {
  stop("Trained model not found at: ", MODEL_PATH,
       "\n\nSteps to get it:",
       "\n  1. Run R/10_export_training_data.R to create data/training/",
       "\n  2. Upload data/training/ to Google Drive",
       "\n  3. Open notebooks/05_train_3d_unet_colab.Rmd in Colab",
       "\n  4. Train the model (takes ~30-60 min on GPU)",
       "\n  5. Download unet_3d_trained.pt to models/")
}

device <- if (cuda_is_available()) "cuda" else "cpu"
cat("Device:", device, "\n")

unet_model <- load_trained_unet(MODEL_PATH, device)
if (is.null(unet_model)) stop("Failed to load model")

# ---- Load dataset ----

cat("\n==== Loading Dataset ====\n")
summary_data <- summarize_dataset(DATA_PATH)
patients <- summary_data$patients

N_PATIENTS <- 15
TARGET_SIZE <- c(48L, 48L, 48L)

selected <- patients %>%
  filter(n_timepoints >= 3) %>%
  arrange(desc(n_timepoints)) %>%
  head(N_PATIENTS)

cat("Processing", nrow(selected), "patients\n")

# ---- Process each patient ----

all_region_tables <- list()
all_event_tables <- list()
all_progressions <- list()
all_trajectories <- list()
seg_dice_scores <- list()

for (i in seq_len(nrow(selected))) {
  pid <- selected$patient_id[i]
  cat("\n[", i, "/", nrow(selected), "]", pid, "\n")

  patient_data <- tryCatch(
    load_full_patient(selected$path[i]),
    error = function(e) { cat("  SKIP:", e$message, "\n"); NULL }
  )
  if (is.null(patient_data)) next

  preprocessed <- tryCatch(
    preprocess_full_patient(patient_data, target_size = TARGET_SIZE),
    error = function(e) { cat("  FAIL:", e$message, "\n"); NULL }
  )
  if (is.null(preprocessed)) next

  # Segment with trained U-Net
  cat("  Segmenting with trained U-Net...\n")
  segmented <- segment_patient(unet_model, preprocessed, device)

  # Compare with ground truth (dataset segmentation)
  for (tp_name in names(segmented)) {
    gt_seg <- preprocessed[[tp_name]]$segmentation
    pred_seg <- segmented[[tp_name]]$segmentation
    if (!is.null(gt_seg) && !is.null(pred_seg)) {
      dice <- compute_dice(gt_seg > 0, pred_seg > 0)
      seg_dice_scores[[paste0(pid, "_", tp_name)]] <- dice
    }
  }

  # Extract regions from U-Net predictions
  region_data <- extract_all_patient_regions(segmented, pid, min_vol = 50)
  all_region_tables[[pid]] <- region_data$table

  if (length(region_data$regions_by_timepoint) < 2) next

  # Track regions
  tracking <- track_patient_regions(region_data$regions_by_timepoint)
  events <- create_event_table(tracking$all_events, pid)
  all_event_tables[[pid]] <- events

  # Trajectories
  tp_names <- names(region_data$regions_by_timepoint)
  time_weeks <- sapply(patient_data$timepoints[tp_names], `[[`, "week_number")

  trajectories <- list()
  for (r in region_data$regions_by_timepoint[[1]]) {
    vols <- sapply(tp_names, function(tp) {
      matched <- Filter(function(x) x$region_type == r$region_type,
                        region_data$regions_by_timepoint[[tp]])
      if (length(matched) > 0) matched[[1]]$volume_cm3 else 0
    })
    trajectories[[r$region_id]] <- analyze_region_trajectory(vols, time_weeks)
  }
  all_trajectories[[pid]] <- trajectories

  # Classify progression
  progression <- classify_patient_progression(trajectories, events)
  all_progressions[[pid]] <- progression
  cat("  Progression:", progression$classification,
      "(confidence:", round(progression$confidence, 2), ")\n")
}

# ---- Aggregate results ----

combined_regions <- bind_rows(all_region_tables)
combined_events <- bind_rows(all_event_tables)

cat("\n========================================\n")
cat("Results Summary\n")
cat("========================================\n")
cat("Patients processed:", length(all_progressions), "\n")
cat("Regions extracted: ", nrow(combined_regions), "\n")
cat("Events detected:   ", nrow(combined_events), "\n")

# ---- U-Net Segmentation Quality ----

cat("\n==== 3D U-Net Segmentation Quality ====\n")
dice_vals <- unlist(seg_dice_scores)
cat("Mean Dice (whole tumor):", round(mean(dice_vals), 4), "\n")
cat("Std Dice:               ", round(sd(dice_vals), 4), "\n")
cat("Min Dice:               ", round(min(dice_vals), 4), "\n")
cat("Max Dice:               ", round(max(dice_vals), 4), "\n")

p_dice <- tibble(dice = dice_vals) %>%
  ggplot(aes(dice)) +
  geom_histogram(binwidth = 0.05, fill = "#2563eb", color = "white", alpha = 0.85) +
  geom_vline(xintercept = mean(dice_vals), color = "#dc2626",
             linetype = "dashed", linewidth = 1) +
  annotate("text", x = mean(dice_vals) + 0.02, y = Inf, vjust = 2,
           label = paste("Mean:", round(mean(dice_vals), 3)),
           color = "#dc2626", fontface = "bold") +
  labs(title = "3D U-Net Segmentation — Dice Score Distribution",
       subtitle = paste(length(dice_vals), "timepoint predictions vs ground truth"),
       x = "Dice Score", y = "Count") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))
ggsave(file.path(FIGURES_PATH, "13_unet_dice_scores.png"),
       p_dice, width = 8, height = 5, dpi = 150)

# ---- Progression Classification ----

cat("\n==== Progression Classification ====\n")
prog_df <- tibble(
  patient_id = names(all_progressions),
  classification = sapply(all_progressions, `[[`, "classification"),
  confidence = sapply(all_progressions, `[[`, "confidence")
)
print(table(prog_df$classification))

prog_colors <- c("CR" = "#2ecc71", "PR" = "#3498db", "SD" = "#f39c12", "PD" = "#e74c3c")

p_prog <- prog_df %>%
  mutate(classification = factor(classification, levels = c("CR", "PR", "SD", "PD"))) %>%
  count(classification, .drop = FALSE) %>%
  ggplot(aes(classification, n, fill = classification)) +
  geom_col(alpha = 0.85, width = 0.6) +
  geom_text(aes(label = n), vjust = -0.5, fontface = "bold", size = 5) +
  labs(title = "Progression Classification (Trained U-Net + LSTM)",
       subtitle = paste(nrow(prog_df), "patients"),
       x = "RANO Category", y = "Patient Count") +
  scale_fill_manual(values = prog_colors) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none", plot.title = element_text(face = "bold"))
ggsave(file.path(FIGURES_PATH, "13_progression_trained_model.png"),
       p_prog, width = 8, height = 5, dpi = 150)

# ---- Volume trajectories ----

if (nrow(combined_regions) > 0) {
  top_pids <- names(all_progressions)[seq_len(min(6, length(all_progressions)))]

  p_traj <- combined_regions %>%
    filter(patient_id %in% top_pids) %>%
    ggplot(aes(x = timepoint, y = volume_cm3,
               color = region_type, group = region_id)) +
    geom_line(linewidth = 1.2) +
    geom_point(size = 2.5) +
    facet_wrap(~patient_id, scales = "free", ncol = 3) +
    labs(title = "Region Volume Trajectories (Trained U-Net Segmentation)",
         x = "Timepoint", y = "Volume (cm3)", color = "Region Type") +
    scale_color_brewer(palette = "Set1") +
    theme_minimal(base_size = 12) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7),
          plot.title = element_text(face = "bold"),
          strip.text = element_text(face = "bold"))
  ggsave(file.path(FIGURES_PATH, "13_volume_trajectories_trained.png"),
         p_traj, width = 14, height = 9, dpi = 150)
}

# ---- LSTM Training ----

cat("\n==== Training LSTM ====\n")

lstm_sequences <- list()
lstm_labels <- c()
long_features <- compute_longitudinal_features(combined_regions, combined_events)

for (pid in names(all_progressions)) {
  pt_regions <- combined_regions %>% filter(patient_id == pid)
  pt_events <- combined_events %>% filter(patient_id == pid)
  timepoints <- sort(unique(pt_regions$timepoint))
  if (length(timepoints) < 2) next

  MAX_SEQ <- 20L; N_FEAT <- 13L
  seq_mat <- matrix(0, nrow = MAX_SEQ, ncol = N_FEAT)

  for (t_idx in seq_along(timepoints)) {
    if (t_idx > MAX_SEQ) break
    tp <- timepoints[t_idx]
    tp_data <- pt_regions %>% filter(timepoint == tp)

    total_vol <- sum(tp_data$volume_cm3, na.rm = TRUE)
    n_regions <- nrow(tp_data)
    max_vol <- max(tp_data$volume_cm3, na.rm = TRUE)
    mean_vol <- mean(tp_data$volume_cm3, na.rm = TRUE)

    vol_change <- 0; vol_pct <- 0; max_growth <- 0; max_disp <- 0
    n_grow <- 0; n_appear <- 0; n_disappear <- 0; n_split <- 0; n_merge <- 0

    if (t_idx > 1) {
      prev_tp <- timepoints[t_idx - 1]
      prev_vol <- pt_regions %>%
        filter(timepoint == prev_tp) %>%
        summarise(v = sum(volume_cm3, na.rm = TRUE)) %>% pull(v)
      vol_change <- total_vol - prev_vol
      vol_pct <- if (prev_vol > 0) vol_change / prev_vol else 0

      tp_ev <- pt_events %>% filter(to_tp == tp)
      if (nrow(tp_ev) > 0) {
        n_appear <- sum(tp_ev$event_type == "APPEARANCE")
        n_disappear <- sum(tp_ev$event_type == "DISAPPEARANCE")
        n_split <- sum(tp_ev$event_type == "SPLIT")
        n_merge <- sum(tp_ev$event_type == "MERGE")
      }

      tp_lf <- long_features %>% filter(patient_id == pid, to_tp == tp)
      if (nrow(tp_lf) > 0) {
        max_growth <- max(tp_lf$pct_volume_change, na.rm = TRUE)
        max_disp <- max(tp_lf$centroid_displacement, na.rm = TRUE)
        n_grow <- sum(tp_lf$pct_volume_change > 0.25, na.rm = TRUE)
      }
    }

    seq_mat[t_idx, ] <- c(total_vol, n_regions, max_vol, mean_vol,
                            vol_change, vol_pct, max_growth, max_disp,
                            n_grow, n_appear, n_disappear, n_split, n_merge)
  }

  lstm_sequences <- c(lstm_sequences, list(seq_mat))
  prog <- all_progressions[[pid]]$classification
  lstm_labels <- c(lstm_labels, match(prog, c("CR", "PR", "SD", "PD")) - 1L)
}

cat("LSTM sequences:", length(lstm_sequences), "\n")
print(table(c("CR", "PR", "SD", "PD")[lstm_labels + 1]))

lstm_result <- NULL
if (length(lstm_sequences) >= 4) {
  set.seed(42)
  n <- length(lstm_sequences)
  train_idx <- sample(n, max(floor(n * 0.8), 3))
  val_idx <- setdiff(seq_len(n), train_idx)
  if (length(val_idx) == 0) val_idx <- train_idx[seq_len(min(2, length(train_idx)))]

  lstm_result <- train_lstm(
    train_sequences = lstm_sequences[train_idx],
    train_labels    = lstm_labels[train_idx],
    val_sequences   = lstm_sequences[val_idx],
    val_labels      = lstm_labels[val_idx],
    input_size  = 13L, hidden_size = 64L, num_layers = 2L,
    n_classes   = 4L, epochs = 50L, batch_size = 4L, lr = 1e-3,
    save_path   = "models/lstm_progression.pt"
  )

  if (length(lstm_result$history$train_loss) > 0) {
    p_loss <- tibble(
      epoch = seq_along(lstm_result$history$train_loss),
      Train = lstm_result$history$train_loss,
      Val   = lstm_result$history$val_loss
    ) %>%
      pivot_longer(-epoch, names_to = "Split", values_to = "Loss") %>%
      ggplot(aes(epoch, Loss, color = Split)) +
      geom_line(linewidth = 1.2) +
      scale_color_manual(values = c(Train = "#2563eb", Val = "#dc2626")) +
      labs(title = "LSTM Training Loss") +
      theme_minimal(base_size = 13) +
      theme(plot.title = element_text(face = "bold"))

    p_acc <- tibble(
      epoch = seq_along(lstm_result$history$train_acc),
      Train = lstm_result$history$train_acc * 100,
      Val   = lstm_result$history$val_acc * 100
    ) %>%
      pivot_longer(-epoch, names_to = "Split", values_to = "Accuracy") %>%
      ggplot(aes(epoch, Accuracy, color = Split)) +
      geom_line(linewidth = 1.2) +
      scale_color_manual(values = c(Train = "#2563eb", Val = "#dc2626")) +
      labs(title = "LSTM Accuracy (%)") +
      theme_minimal(base_size = 13) +
      theme(plot.title = element_text(face = "bold"))

    p_lstm <- p_loss + p_acc +
      plot_annotation(title = "LSTM Progression Model Training")
    ggsave(file.path(FIGURES_PATH, "14_lstm_training_trained.png"),
           p_lstm, width = 14, height = 5, dpi = 150)
  }
}

# ---- Ablation Study ----

cat("\n==== Ablation Study ====\n")
patient_summaries <- compute_patient_summary_features(
  combined_regions, combined_events, long_features
)

rano_labels <- tibble(
  patient_id = names(all_progressions),
  rano_label = sapply(all_progressions, `[[`, "classification")
)

if (nrow(patient_summaries) > 0 && nrow(rano_labels) > 0) {
  ablation <- run_ablation_study(patient_summaries, rano_labels)
  print_validation_report(ablation)

  abl_table <- format_ablation_table(ablation)
  write_csv(abl_table, file.path(TABLES_PATH, "ablation_trained_model.csv"))
}

# ---- Save everything ----

cat("\n==== Saving Results ====\n")
write_csv(combined_regions, file.path(TABLES_PATH, "regions_trained_model.csv"))
write_csv(combined_events, file.path(TABLES_PATH, "events_trained_model.csv"))
write_csv(long_features, file.path(TABLES_PATH, "longitudinal_features_trained.csv"))
write_csv(patient_summaries, file.path(TABLES_PATH, "patient_summaries_trained.csv"))
write_csv(prog_df, file.path(TABLES_PATH, "predictions_trained_model.csv"))
write_csv(tibble(sample = names(seg_dice_scores), dice = unlist(seg_dice_scores)),
          file.path(TABLES_PATH, "unet_dice_scores.csv"))

cat("\n")
cat("============================================================\n")
cat("  FULL PIPELINE COMPLETE (Trained Model)\n")
cat("============================================================\n")
cat("  3D U-Net Dice:         ", round(mean(dice_vals), 4), "\n")
cat("  Patients processed:    ", length(all_progressions), "\n")
cat("  Regions extracted:     ", nrow(combined_regions), "\n")
cat("  Events detected:       ", nrow(combined_events), "\n")
cat("  LSTM trained:          ", !is.null(lstm_result), "\n")
cat("  Figures saved:         ", FIGURES_PATH, "\n")
cat("  Tables saved:          ", TABLES_PATH, "\n")
cat("============================================================\n")
