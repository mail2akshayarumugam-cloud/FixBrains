# ══════════════════════════════════════════════════════════════
# Inference Pipeline — Run on ANY Patient
# Input: folder with MRI scans (NIfTI), no segmentation needed
# The trained 3D U-Net handles segmentation automatically
#
# Usage:
#   Rscript run_inference.R --patient path/to/patient_folder
#   Rscript run_inference.R --input path/to/folder_of_patients
#   Rscript run_inference.R --input data/raw/LUMIERE  (all patients)
# ══════════════════════════════════════════════════════════════

setwd("C:/Users/mail2/OneDrive/Desktop/R project 1")

suppressPackageStartupMessages({
  library(tidyverse)
  library(RNifti)
  library(torch)
  library(ggplot2)
  library(patchwork)
  library(viridis)
})

source("R/utils.R")
source("R/01_data_loader.R")
source("R/02_preprocessing.R")
source("R/03_model_3d_unet.R")
source("R/04_region_extractor.R")
source("R/05_region_tracker.R")
source("R/06_feature_extraction.R")
source("R/07_progression_engine.R")
source("R/11_inference.R")

# ---- Config ----

MODEL_PATH <- "models/unet_3d_trained_jit.pt"
TARGET_SIZE <- c(48L, 48L, 48L)
OUTPUT_DIR <- "results/inference"

args <- commandArgs(trailingOnly = TRUE)

patient_path <- NULL
input_dir <- NULL

i <- 1
while (i <= length(args)) {
  if (args[i] == "--patient") {
    patient_path <- args[i + 1]; i <- i + 2
  } else if (args[i] == "--input") {
    input_dir <- args[i + 1]; i <- i + 2
  } else if (args[i] == "--model") {
    MODEL_PATH <- args[i + 1]; i <- i + 2
  } else if (args[i] == "--output") {
    OUTPUT_DIR <- args[i + 1]; i <- i + 2
  } else {
    i <- i + 1
  }
}

ensure_dir(OUTPUT_DIR)
ensure_dir(file.path(OUTPUT_DIR, "reports"))
ensure_dir(file.path(OUTPUT_DIR, "figures"))

# ---- Load model ----

if (!file.exists(MODEL_PATH)) {
  stop("Trained model not found at: ", MODEL_PATH,
       "\n\nTo get the trained model:",
       "\n  1. Run: Rscript R/10_export_training_data.R",
       "\n  2. Upload data/training/ to Google Drive",
       "\n  3. Train in Colab: notebooks/05_train_3d_unet_colab.Rmd",
       "\n  4. Download model to: ", MODEL_PATH)
}

device <- if (cuda_is_available()) "cuda" else "cpu"
cat("Device:", device, "\n")
cat("Loading trained 3D U-Net...\n")
model <- load_trained_unet(MODEL_PATH, device)
if (is.null(model)) stop("Failed to load model")
cat("Model loaded.\n\n")

# ---- Discover patients ----

if (!is.null(patient_path)) {
  patient_dirs <- patient_path
  patient_ids <- basename(patient_path)
} else if (!is.null(input_dir)) {
  summary_data <- summarize_dataset(input_dir)
  patients <- summary_data$patients %>%
    filter(n_timepoints >= 2) %>%
    arrange(desc(n_timepoints))
  patient_dirs <- patients$path
  patient_ids <- patients$patient_id
  cat("Found", nrow(patients), "patients with 2+ timepoints\n\n")
} else {
  stop("Provide --patient <path> for one patient or --input <dir> for a batch")
}

# ---- Process each patient ----

all_results <- list()
all_region_tables <- list()
all_event_tables <- list()
all_progressions <- list()

for (i in seq_along(patient_dirs)) {
  pid <- patient_ids[i]
  cat("═══════════════════════════════════════\n")
  cat("[", i, "/", length(patient_dirs), "]", pid, "\n")
  cat("═══════════════════════════════════════\n")

  patient_data <- tryCatch(
    load_full_patient(patient_dirs[i]),
    error = function(e) { cat("  SKIP:", e$message, "\n"); NULL }
  )
  if (is.null(patient_data)) next

  preprocessed <- tryCatch(
    preprocess_full_patient(patient_data, target_size = TARGET_SIZE),
    error = function(e) { cat("  PREPROCESS FAIL:", e$message, "\n"); NULL }
  )
  if (is.null(preprocessed)) next

  result <- tryCatch(
    run_full_inference_pipeline(patient_data, preprocessed, MODEL_PATH, device, preloaded_model = model),
    error = function(e) { cat("  INFERENCE FAIL:", e$message, "\n"); NULL }
  )
  if (is.null(result)) next

  all_results[[pid]] <- result
  all_region_tables[[pid]] <- result$regions$table

  if (!is.null(result$events)) all_event_tables[[pid]] <- result$events
  if (!is.null(result$progression)) {
    all_progressions[[pid]] <- result$progression
    cat("  Progression:", result$progression$classification,
        "(confidence:", round(result$progression$confidence, 2), ")\n")
  }

  # Per-patient report
  report_file <- file.path(OUTPUT_DIR, "reports", paste0(pid, "_report.txt"))
  sink(report_file)
  cat("PATIENT:", pid, "\n")
  cat("Timepoints:", length(preprocessed), "\n")
  cat("Regions:", nrow(result$regions$table), "\n")
  if (!is.null(result$events)) cat("Events:", nrow(result$events), "\n")
  if (!is.null(result$progression)) {
    cat("Progression:", result$progression$classification, "\n")
    cat("Confidence:", round(result$progression$confidence, 2), "\n")
    cat("\nReasoning:\n")
    for (r in result$progression$reasons) cat("  -", r, "\n")
  }
  cat("\nRegions:\n")
  print(result$regions$table)
  if (!is.null(result$events) && nrow(result$events) > 0) {
    cat("\nEvents:\n")
    print(result$events)
  }
  sink()
  cat("  Report saved:", report_file, "\n")

  rm(patient_data, preprocessed, result); gc()
}

# ---- Aggregate results ----

combined_regions <- bind_rows(all_region_tables)
combined_events <- bind_rows(all_event_tables)

cat("\n")
cat("══════════════════════════════════════════════\n")
cat("  INFERENCE COMPLETE\n")
cat("══════════════════════════════════════════════\n")
cat("  Patients analyzed:    ", length(all_progressions), "\n")
cat("  Regions found:        ", nrow(combined_regions), "\n")
cat("  Events detected:      ", nrow(combined_events), "\n")
cat("  Output:               ", OUTPUT_DIR, "\n")
cat("══════════════════════════════════════════════\n\n")

# ---- Classification summary ----

if (length(all_progressions) > 0) {
  prog_df <- tibble(
    patient_id = names(all_progressions),
    classification = sapply(all_progressions, `[[`, "classification"),
    confidence = sapply(all_progressions, `[[`, "confidence")
  )

  cat("Progression Summary:\n")
  print(table(prog_df$classification))
  cat("\n")
  print(prog_df)

  write_csv(prog_df, file.path(OUTPUT_DIR, "predictions.csv"))
  write_csv(combined_regions, file.path(OUTPUT_DIR, "all_regions.csv"))
  write_csv(combined_events, file.path(OUTPUT_DIR, "all_events.csv"))

  # Summary figure
  prog_colors <- c("CR" = "#2ecc71", "PR" = "#3498db",
                    "SD" = "#f39c12", "PD" = "#e74c3c")

  p <- prog_df %>%
    mutate(classification = factor(classification,
                                   levels = c("CR", "PR", "SD", "PD"))) %>%
    count(classification, .drop = FALSE) %>%
    ggplot(aes(classification, n, fill = classification)) +
    geom_col(alpha = 0.85, width = 0.6) +
    geom_text(aes(label = n), vjust = -0.5, fontface = "bold", size = 5) +
    labs(title = "Tumor Progression Classification",
         subtitle = paste(nrow(prog_df), "patients — automated U-Net segmentation"),
         x = "RANO Category", y = "Patient Count") +
    scale_fill_manual(values = prog_colors) +
    theme_minimal(base_size = 14) +
    theme(legend.position = "none", plot.title = element_text(face = "bold"))
  ggsave(file.path(OUTPUT_DIR, "figures", "progression_summary.png"),
         p, width = 8, height = 5, dpi = 150)
}

cat("Done. Check", OUTPUT_DIR, "for all outputs.\n")
