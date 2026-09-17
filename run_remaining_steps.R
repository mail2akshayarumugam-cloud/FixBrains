setwd("C:/Users/mail2/OneDrive/Desktop/R project 1")

library(tidyverse)
library(data.table)
library(ggplot2)
library(patchwork)
library(viridis)
library(torch)
library(caret)
library(irr)
library(pROC)

source("R/utils.R")
source("R/08_model_lstm.R")
source("R/09_clinical_validation.R")

MODELS_PATH <- "models"
RESULTS_PATH <- "results"
FIGURES_PATH <- file.path(RESULTS_PATH, "figures")
TABLES_PATH <- file.path(RESULTS_PATH, "tables")
ensure_dir(MODELS_PATH)

cat("Loading saved data from previous run...\n")
combined_regions <- read_csv(file.path(TABLES_PATH, "all_regions.csv"), show_col_types = FALSE)
combined_events  <- read_csv(file.path(TABLES_PATH, "all_events.csv"), show_col_types = FALSE)
long_features    <- read_csv(file.path(TABLES_PATH, "longitudinal_features.csv"), show_col_types = FALSE)
patient_summaries <- read_csv(file.path(TABLES_PATH, "patient_summaries.csv"), show_col_types = FALSE)
rano_labels      <- read_csv(file.path(TABLES_PATH, "predictions.csv"), show_col_types = FALSE)

all_progressions <- split(rano_labels, rano_labels$patient_id) %>%
  lapply(function(x) list(classification = x$rano_label[1]))

cat("Loaded:", nrow(combined_regions), "regions,", nrow(combined_events), "events,",
    length(all_progressions), "patients\n\n")

# ---- STEP 9: LSTM ----
tryCatch({
cat("══════════════════════════════════════════\n")
cat("  STEP 9: LSTM Training\n")
cat("══════════════════════════════════════════\n")

lstm_sequences <- list()
lstm_labels <- c()

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
      n_appear <- sum(tp_ev$event_type == "APPEARANCE")
      n_disappear <- sum(tp_ev$event_type == "DISAPPEARANCE")
      n_split <- sum(tp_ev$event_type == "SPLIT")
      n_merge <- sum(tp_ev$event_type == "MERGE")

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
  lstm_labels <- c(lstm_labels, match(prog, c("CR", "PR", "SD", "PD")))
}

cat("LSTM sequences:", length(lstm_sequences), "\n")
cat("Label distribution:\n")
if (length(lstm_labels) > 0) {
  print(table(c("CR", "PR", "SD", "PD")[lstm_labels]))
}

lstm_result <- NULL
if (length(lstm_sequences) >= 4) {
  set.seed(42)
  n <- length(lstm_sequences)
  train_idx <- sample(n, max(floor(n * 0.8), 3))
  val_idx <- setdiff(seq_len(n), train_idx)
  if (length(val_idx) == 0) val_idx <- train_idx[1:min(2, length(train_idx))]

  cat("Train:", length(train_idx), "| Val:", length(val_idx), "\n")

  lstm_result <- train_lstm(
    train_sequences = lstm_sequences[train_idx],
    train_labels    = lstm_labels[train_idx],
    val_sequences   = lstm_sequences[val_idx],
    val_labels      = lstm_labels[val_idx],
    input_size  = 13L, hidden_size = 64L, num_layers = 2L,
    n_classes   = 4L, epochs = 50L, batch_size = 4L, lr = 1e-3,
    save_path   = file.path(MODELS_PATH, "lstm_progression.pt")
  )

  if (length(lstm_result$history$train_loss) > 0) {
    p18a <- tibble(
      epoch = seq_along(lstm_result$history$train_loss),
      Train = lstm_result$history$train_loss,
      Val   = lstm_result$history$val_loss
    ) %>%
      pivot_longer(-epoch, names_to = "Split", values_to = "Loss") %>%
      ggplot(aes(epoch, Loss, color = Split)) +
      geom_line(linewidth = 1.2) + geom_point(size = 1, alpha = 0.5) +
      labs(title = "LSTM Training Loss") +
      scale_color_manual(values = c(Train = "#2563eb", Val = "#dc2626")) +
      theme_minimal(base_size = 13) + theme(plot.title = element_text(face = "bold"))

    p18b <- tibble(
      epoch = seq_along(lstm_result$history$train_acc),
      Train = lstm_result$history$train_acc * 100,
      Val   = lstm_result$history$val_acc * 100
    ) %>%
      pivot_longer(-epoch, names_to = "Split", values_to = "Accuracy") %>%
      ggplot(aes(epoch, Accuracy, color = Split)) +
      geom_line(linewidth = 1.2) + geom_point(size = 1, alpha = 0.5) +
      labs(title = "LSTM Accuracy (%)") +
      scale_color_manual(values = c(Train = "#2563eb", Val = "#dc2626")) +
      theme_minimal(base_size = 13) + theme(plot.title = element_text(face = "bold"))

    p18 <- p18a + p18b + plot_annotation(
      title = "LSTM Temporal Progression Model — Training",
      subtitle = "Input: 13-feature temporal sequences | Output: RANO class",
      theme = theme(plot.title = element_text(face = "bold", size = 16))
    )
    print(p18)
    ggsave(file.path(FIGURES_PATH, "10_lstm_training.png"), p18, width = 14, height = 5, dpi = 150)
  }
}

cat("Step 9 complete.\n")
}, error = function(e) cat("STEP 9 ERROR:", e$message, "\n"))

# ---- STEP 10: Ablation ----
tryCatch({
cat("\n══════════════════════════════════════════\n")
cat("  STEP 10: Ablation Study\n")
cat("══════════════════════════════════════════\n")

ablation <- run_ablation_study(patient_summaries, rano_labels)
print_validation_report(ablation)

abl_table <- format_ablation_table(ablation)
write_csv(abl_table, file.path(TABLES_PATH, "ablation_results.csv"))

p19 <- abl_table %>%
  mutate(across(c(`Macro F1`, Kappa, `PD Sensitivity`, `PD F1`), ~as.numeric(.))) %>%
  pivot_longer(c(`Macro F1`, Kappa, `PD Sensitivity`, `PD F1`),
               names_to = "Metric", values_to = "Value") %>%
  ggplot(aes(Approach, Value, fill = Metric)) +
  geom_col(position = "dodge", alpha = 0.85) +
  geom_text(aes(label = sprintf("%.2f", Value)),
            position = position_dodge(width = 0.9), vjust = -0.5, size = 3) +
  labs(title = "Ablation Study: Total Volume vs Region-Wise vs Full Framework",
       subtitle = "Comparing three approaches against clinical RANO criteria",
       y = "Score", x = "") +
  scale_fill_brewer(palette = "Set2") +
  theme_minimal(base_size = 13) +
  theme(axis.text.x = element_text(angle = 15, hjust = 1),
        plot.title = element_text(face = "bold"))
print(p19)
ggsave(file.path(FIGURES_PATH, "11_ablation_comparison.png"), p19, width = 12, height = 6, dpi = 150)

for (approach_name in names(ablation)) {
  r <- ablation[[approach_name]]
  cm <- as.data.frame(r$metrics$confusion_matrix)
  names(cm) <- c("Predicted", "Actual", "Count")
  p_cm <- ggplot(cm, aes(x = Actual, y = Predicted, fill = Count)) +
    geom_tile(color = "white", linewidth = 1) +
    geom_text(aes(label = Count), size = 6, fontface = "bold") +
    scale_fill_gradient(low = "white", high = "#2563eb") +
    labs(title = paste("Confusion Matrix —", r$name)) +
    theme_minimal(base_size = 13) + theme(plot.title = element_text(face = "bold"))
  print(p_cm)
  ggsave(file.path(FIGURES_PATH, paste0("11_confusion_", approach_name, ".png")),
         p_cm, width = 6, height = 5, dpi = 150)
}

cat("Step 10 complete.\n")
}, error = function(e) cat("STEP 10 ERROR:", e$message, "\n"))

# ---- Summary ----
cat("\n══════════════════════════════════════════════════════════════\n")
cat("  REMAINING STEPS COMPLETE\n")
cat("══════════════════════════════════════════════════════════════\n")
cat("  LSTM trained:  ", if (!is.null(lstm_result)) "Yes" else "No", "\n")
cat("\nFigures:\n")
for (f in list.files(FIGURES_PATH, pattern = "\\.png$")) cat("  ", f, "\n")
cat("\nTables:\n")
for (t in list.files(TABLES_PATH, pattern = "\\.csv$")) cat("  ", t, "\n")
cat("══════════════════════════════════════════════════════════════\n")
