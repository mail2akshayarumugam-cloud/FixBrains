setwd("C:/Users/mail2/OneDrive/Desktop/R project 1")

library(tidyverse)
library(torch)
library(ggplot2)
library(patchwork)
library(viridis)

MODELS_PATH  <- "models"
RESULTS_PATH <- "results"
FIGURES_PATH <- file.path(RESULTS_PATH, "figures")
TABLES_PATH  <- file.path(RESULTS_PATH, "tables")
TRAINING_DIR <- "data/training"

cat("══════════════════════════════════════════\n")
cat("  Integrating 3D U-Net into Pipeline\n")
cat("══════════════════════════════════════════\n\n")

# ── 1. Copy Colab figures to results ─────────────────────────
tryCatch({
  cat("[1/4] Copying Colab training figures...\n")
  file.copy(file.path(MODELS_PATH, "training_curves.png"),
            file.path(FIGURES_PATH, "13_unet_training_curves.png"), overwrite = TRUE)
  file.copy(file.path(MODELS_PATH, "visual_results.png"),
            file.path(FIGURES_PATH, "14_unet_visual_results.png"), overwrite = TRUE)
  cat("  Copied to results/figures/\n")
}, error = function(e) cat("  ERROR copying figures:", e$message, "\n"))

# ── 2. U-Net Dice score bar chart ────────────────────────────
tryCatch({
  cat("[2/4] Creating U-Net summary figure...\n")
  unet_results <- read_csv(file.path(MODELS_PATH, "unet_results.csv"), show_col_types = FALSE)

  dice_data <- tibble(
    Region = c("Necrosis", "Edema", "Enhancing", "Overall"),
    Dice = c(0.0243, 0.2183, 0.4007, 0.2230)
  ) %>% mutate(Region = factor(Region, levels = c("Necrosis", "Edema", "Enhancing", "Overall")))

  p_dice <- ggplot(dice_data, aes(Region, Dice, fill = Region)) +
    geom_col(alpha = 0.85, width = 0.6) +
    geom_text(aes(label = sprintf("%.3f", Dice)), vjust = -0.5, fontface = "bold", size = 5) +
    scale_fill_manual(values = c("Necrosis" = "#e74c3c", "Edema" = "#f39c12",
                                  "Enhancing" = "#2ecc71", "Overall" = "#3498db")) +
    labs(title = "3D U-Net Segmentation — Per-Class Dice Scores",
         subtitle = "Trained on 599 samples (91 patients) | 100 epochs | T4 GPU",
         y = "Dice Score", x = "") +
    ylim(0, 0.55) +
    theme_minimal(base_size = 14) +
    theme(legend.position = "none", plot.title = element_text(face = "bold"))

  # Architecture info panel
  arch_info <- tibble(
    Property = c("Input", "Output", "Parameters", "Encoder", "Bottleneck",
                 "Loss Function", "Optimizer", "Best Val Dice"),
    Value = c("4-ch MRI (48³)", "4-class seg", "5,649,204", "4 levels (16→128)",
              "256 filters", "Dice + CE", "Adam (lr=1e-4)", "0.2733")
  )

  p_arch <- ggplot(arch_info, aes(x = 0, y = rev(seq_len(nrow(arch_info))))) +
    geom_text(aes(label = paste0(Property, ": ", Value)),
              hjust = 0, size = 4, fontface = "plain") +
    xlim(-0.1, 3) +
    labs(title = "3D U-Net Architecture Summary") +
    theme_void() +
    theme(plot.title = element_text(face = "bold", size = 14, hjust = 0.5))

  p_combined <- p_dice + p_arch +
    plot_layout(widths = c(2, 1)) +
    plot_annotation(
      title = "3D U-Net for Tumor Segmentation — LUMIERE Dataset",
      theme = theme(plot.title = element_text(face = "bold", size = 16))
    )
  print(p_combined)
  ggsave(file.path(FIGURES_PATH, "13_unet_dice_summary.png"),
         p_combined, width = 14, height = 6, dpi = 150)
  cat("  Saved 13_unet_dice_summary.png\n")
}, error = function(e) cat("  ERROR creating summary figure:", e$message, "\n"))

# ── 3. Run inference on validation samples ───────────────────
tryCatch({
  cat("[3/4] Running U-Net inference on sample patients...\n")

  model <- jit_load(file.path(MODELS_PATH, "unet_3d_trained_jit.pt"))
  cat("  JIT model loaded\n")

  manifest <- read_csv(file.path(TRAINING_DIR, "manifest.csv"), show_col_types = FALSE) %>%
    filter(tumor_voxels > 0)

  set.seed(42)
  patients <- unique(manifest$patient_id)
  n_train <- floor(length(patients) * 0.8)
  val_patients <- patients[(n_train + 1):length(patients)]

  val_manifest <- manifest %>% filter(patient_id %in% val_patients)
  cat("  Validation samples:", nrow(val_manifest), "\n")

  # Run inference on up to 8 samples, show 4
  n_infer <- min(8, nrow(val_manifest))
  infer_samples <- val_manifest[1:n_infer, ]

  dice_results <- list()
  plot_data <- list()

  for (i in seq_len(n_infer)) {
    row <- infer_samples[i, ]
    mri_path <- file.path(TRAINING_DIR, row$mri_file)
    seg_path <- file.path(TRAINING_DIR, row$seg_file)

    mri <- torch_load(mri_path)$to(dtype = torch_float())$unsqueeze(1)
    seg_gt <- torch_load(seg_path)$to(dtype = torch_long())

    with_no_grad({
      pred <- model(mri)
      pred_class <- pred$argmax(dim = 2)$squeeze(1)
    })

    seg_gt_arr <- as.array(seg_gt$cpu())
    pred_arr <- as.array(pred_class$cpu()) - 1L

    class_dice <- c()
    for (c in 1:3) {
      pred_c <- (pred_arr == c)
      gt_c <- (seg_gt_arr == c)
      intersection <- sum(pred_c & gt_c)
      union_val <- sum(pred_c) + sum(gt_c)
      dice <- if (union_val > 0) 2 * intersection / union_val else NA_real_
      class_dice <- c(class_dice, dice)
    }

    mean_dice <- mean(class_dice, na.rm = TRUE)
    dice_results[[i]] <- tibble(
      patient_id = row$patient_id, timepoint = row$timepoint,
      dice_necrosis = class_dice[1], dice_edema = class_dice[2],
      dice_enhancing = class_dice[3], dice_mean = mean_dice
    )

    cat("  ", row$patient_id, row$timepoint, "— Dice:", round(mean_dice, 4), "\n")

    if (i <= 4) {
      z <- dim(seg_gt_arr)[3] %/% 2
      mri_arr <- as.array(mri$squeeze(1)$cpu())

      plot_data[[i]] <- list(
        patient = paste(row$patient_id, row$timepoint),
        flair = mri_arr[1, , , z],
        gt = seg_gt_arr[, , z],
        pred = pred_arr[, , z],
        dice = mean_dice
      )
    }

    rm(mri, seg_gt, pred, pred_class); gc()
  }

  # Save inference results
  inference_df <- bind_rows(dice_results)
  write_csv(inference_df, file.path(TABLES_PATH, "unet_inference_results.csv"))
  cat("  Saved unet_inference_results.csv\n")

  # Create comparison figure
  if (length(plot_data) >= 1) {
    n_show <- min(4, length(plot_data))
    seg_colors <- c("0" = "black", "1" = "#e74c3c", "2" = "#f39c12", "3" = "#2ecc71")

    plots <- list()
    for (i in seq_len(n_show)) {
      pd <- plot_data[[i]]

      df_flair <- expand.grid(x = seq_len(nrow(pd$flair)), y = seq_len(ncol(pd$flair)))
      df_flair$val <- as.vector(pd$flair)

      df_gt <- expand.grid(x = seq_len(nrow(pd$gt)), y = seq_len(ncol(pd$gt)))
      df_gt$val <- factor(as.vector(pd$gt))

      df_pred <- expand.grid(x = seq_len(nrow(pd$pred)), y = seq_len(ncol(pd$pred)))
      df_pred$val <- factor(as.vector(pd$pred))

      p_f <- ggplot(df_flair, aes(x, y, fill = val)) + geom_raster() +
        scale_fill_viridis_c(option = "inferno") + coord_fixed() +
        labs(title = pd$patient) + theme_void() + theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5, size = 9, face = "bold"))

      p_g <- ggplot(df_gt, aes(x, y, fill = val)) + geom_raster() +
        scale_fill_manual(values = seg_colors, drop = FALSE) + coord_fixed() +
        labs(title = "Ground Truth") + theme_void() + theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5, size = 9))

      p_p <- ggplot(df_pred, aes(x, y, fill = val)) + geom_raster() +
        scale_fill_manual(values = seg_colors, drop = FALSE) + coord_fixed() +
        labs(title = paste0("Predicted (", round(pd$dice, 3), ")")) +
        theme_void() + theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5, size = 9))

      plots <- c(plots, list(p_f, p_g, p_p))
    }

    p_inference <- wrap_plots(plots, ncol = 3, nrow = n_show) +
      plot_annotation(
        title = "3D U-Net Inference: FLAIR | Ground Truth | Prediction",
        subtitle = "Axial slice at z=24 for validation patients",
        theme = theme(plot.title = element_text(face = "bold", size = 14))
      )
    print(p_inference)
    ggsave(file.path(FIGURES_PATH, "15_unet_inference_comparison.png"),
           p_inference, width = 10, height = 3.5 * n_show, dpi = 150)
    cat("  Saved 15_unet_inference_comparison.png\n")
  }
}, error = function(e) cat("  ERROR in inference:", e$message, "\n"))

# ── 4. Final comprehensive summary ──────────────────────────
tryCatch({
  cat("[4/4] Creating final summary...\n")

  # Load all result tables
  patient_summaries <- read_csv(file.path(TABLES_PATH, "patient_summaries.csv"), show_col_types = FALSE)
  all_regions <- read_csv(file.path(TABLES_PATH, "all_regions.csv"), show_col_types = FALSE)
  all_events <- read_csv(file.path(TABLES_PATH, "all_events.csv"), show_col_types = FALSE)
  predictions <- read_csv(file.path(TABLES_PATH, "predictions.csv"), show_col_types = FALSE)
  hidden <- read_csv(file.path(TABLES_PATH, "hidden_changes_analysis.csv"), show_col_types = FALSE)

  ablation <- NULL
  abl_path <- file.path(TABLES_PATH, "ablation_results.csv")
  if (file.exists(abl_path)) ablation <- read_csv(abl_path, show_col_types = FALSE)

  inference <- NULL
  inf_path <- file.path(TABLES_PATH, "unet_inference_results.csv")
  if (file.exists(inf_path)) inference <- read_csv(inf_path, show_col_types = FALSE)

  lstm_exists <- file.exists(file.path(MODELS_PATH, "lstm_progression.pt"))

  # Build summary
  summary_rows <- list(
    tibble(Category = "Dataset", Metric = "Total Patients (LUMIERE)", Value = "91"),
    tibble(Category = "Dataset", Metric = "Patients Analyzed (3+ timepoints)", Value = as.character(nrow(patient_summaries))),
    tibble(Category = "Dataset", Metric = "Total Timepoints Exported", Value = "599"),
    tibble(Category = "Regions", Metric = "Total Regions Extracted", Value = as.character(nrow(all_regions))),
    tibble(Category = "Regions", Metric = "Total Events Tracked", Value = as.character(nrow(all_events))),
    tibble(Category = "Regions", Metric = "Hidden Change Cases", Value = as.character(sum(hidden$disagree, na.rm = TRUE))),
    tibble(Category = "Progression", Metric = "Patients Classified", Value = as.character(nrow(predictions))),
    tibble(Category = "Progression", Metric = "PD (Progressive Disease)", Value = as.character(sum(predictions$rano_label == "PD"))),
    tibble(Category = "Progression", Metric = "PR (Partial Response)", Value = as.character(sum(predictions$rano_label == "PR"))),
    tibble(Category = "Progression", Metric = "CR (Complete Response)", Value = as.character(sum(predictions$rano_label == "CR"))),
    tibble(Category = "Progression", Metric = "SD (Stable Disease)", Value = as.character(sum(predictions$rano_label == "SD")))
  )

  summary_rows <- c(summary_rows, list(
    tibble(Category = "3D U-Net", Metric = "Parameters", Value = "5,649,204"),
    tibble(Category = "3D U-Net", Metric = "Training Samples", Value = "599"),
    tibble(Category = "3D U-Net", Metric = "Epochs Trained", Value = "100"),
    tibble(Category = "3D U-Net", Metric = "Best Val Dice (Overall)", Value = "0.2733"),
    tibble(Category = "3D U-Net", Metric = "Dice - Necrosis", Value = "0.0243"),
    tibble(Category = "3D U-Net", Metric = "Dice - Edema", Value = "0.2183"),
    tibble(Category = "3D U-Net", Metric = "Dice - Enhancing", Value = "0.4007")
  ))

  if (!is.null(inference)) {
    summary_rows <- c(summary_rows, list(
      tibble(Category = "3D U-Net", Metric = "Inference Mean Dice",
             Value = sprintf("%.4f", mean(inference$dice_mean, na.rm = TRUE)))
    ))
  }

  summary_rows <- c(summary_rows, list(
    tibble(Category = "LSTM", Metric = "Model Trained", Value = if (lstm_exists) "Yes" else "No"),
    tibble(Category = "LSTM", Metric = "Input Features", Value = "13 temporal features"),
    tibble(Category = "LSTM", Metric = "Architecture", Value = "2-layer, 64 hidden units")
  ))

  if (!is.null(ablation)) {
    for (j in seq_len(nrow(ablation))) {
      row <- ablation[j, ]
      summary_rows <- c(summary_rows, list(
        tibble(Category = "Ablation", Metric = paste(row$Approach, "— Accuracy"),
               Value = as.character(row$Accuracy)),
        tibble(Category = "Ablation", Metric = paste(row$Approach, "— PD F1"),
               Value = as.character(row$`PD F1`))
      ))
    }
  }

  n_figs <- length(list.files(FIGURES_PATH, pattern = "\\.png$"))
  n_tables <- length(list.files(TABLES_PATH, pattern = "\\.csv$"))
  summary_rows <- c(summary_rows, list(
    tibble(Category = "Output", Metric = "Total Figures", Value = as.character(n_figs)),
    tibble(Category = "Output", Metric = "Total Tables", Value = as.character(n_tables))
  ))

  final_summary <- bind_rows(summary_rows)
  write_csv(final_summary, file.path(TABLES_PATH, "final_summary.csv"))
  cat("  Saved final_summary.csv\n")

  # Print summary
  cat("\n══════════════════════════════════════════════════════════════\n")
  cat("  FINAL PROJECT SUMMARY\n")
  cat("══════════════════════════════════════════════════════════════\n")
  for (cat_name in unique(final_summary$Category)) {
    cat("\n  [", cat_name, "]\n")
    sub <- final_summary %>% filter(Category == cat_name)
    for (k in seq_len(nrow(sub))) {
      cat("    ", sub$Metric[k], ":", sub$Value[k], "\n")
    }
  }
  cat("\n══════════════════════════════════════════════════════════════\n")

}, error = function(e) cat("  ERROR creating summary:", e$message, "\n"))

cat("\nIntegration complete.\n")
