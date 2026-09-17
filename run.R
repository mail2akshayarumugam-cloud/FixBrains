# ──────────────────────────────────────────────────────────────
# Main Entry Point: Tumor Progression Analysis Pipeline
# Usage: Rscript run.R [--patient PATIENT_ID] [--all] [--preprocess]
# ──────────────────────────────────────────────────────────────

cat("
╔══════════════════════════════════════════════════════════════╗
║  Region-Wise Longitudinal Tumor Progression Analysis        ║
║  LUMIERE Dataset | 3D U-Net + LSTM | R torch                ║
╚══════════════════════════════════════════════════════════════╝
\n")

# ── Setup ─────────────────────────────────────────────────────

PROJECT_ROOT <- normalizePath(".", winslash = "/")

source(file.path(PROJECT_ROOT, "R", "utils.R"))
source(file.path(PROJECT_ROOT, "R", "01_data_loader.R"))
source(file.path(PROJECT_ROOT, "R", "02_preprocessing.R"))
source(file.path(PROJECT_ROOT, "R", "04_region_extractor.R"))
source(file.path(PROJECT_ROOT, "R", "05_region_tracker.R"))
source(file.path(PROJECT_ROOT, "R", "06_feature_extraction.R"))
source(file.path(PROJECT_ROOT, "R", "07_progression_engine.R"))
source(file.path(PROJECT_ROOT, "R", "09_clinical_validation.R"))

PATHS <- list(
  raw_data   = file.path(PROJECT_ROOT, "data", "raw", "LUMIERE"),
  processed  = file.path(PROJECT_ROOT, "data", "processed"),
  registered = file.path(PROJECT_ROOT, "data", "processed", "registered"),
  features   = file.path(PROJECT_ROOT, "data", "processed", "features"),
  regions    = file.path(PROJECT_ROOT, "data", "processed", "region_tables"),
  models     = file.path(PROJECT_ROOT, "models"),
  results    = file.path(PROJECT_ROOT, "results"),
  figures    = file.path(PROJECT_ROOT, "results", "figures"),
  tables     = file.path(PROJECT_ROOT, "results", "tables"),
  reports    = file.path(PROJECT_ROOT, "results", "patient_reports")
)

for (p in PATHS) ensure_dir(p)

# ── Parse Arguments ───────────────────────────────────────────

args <- commandArgs(trailingOnly = TRUE)
run_all <- "--all" %in% args
run_preprocess <- "--preprocess" %in% args
patient_arg <- NULL
if ("--patient" %in% args) {
  idx <- which(args == "--patient")
  if (idx < length(args)) patient_arg <- args[idx + 1]
}

# ── Step 1: Discover Dataset ─────────────────────────────────

log_msg("Step 1: Discovering dataset...")

if (!dir.exists(PATHS$raw_data)) {
  stop("LUMIERE dataset not found at: ", PATHS$raw_data,
       "\nPlace the downloaded dataset in data/raw/LUMIERE/")
}

dataset_summary <- summarize_dataset(PATHS$raw_data)
patient_index <- dataset_summary$patients
clinical_data <- dataset_summary$clinical

if (nrow(patient_index) == 0) {
  stop("No patients found in dataset. Check the folder structure.")
}

# ── Step 2: Select Patients ───────────────────────────────────

if (!is.null(patient_arg)) {
  target_patients <- patient_arg
} else if (run_all) {
  target_patients <- patient_index$patient_id
} else {
  target_patients <- head(patient_index$patient_id, 5)
  log_msg("Processing first 5 patients. Use --all for all patients.")
}

log_msg("Processing ", length(target_patients), " patients")

# ── Step 3: Process Each Patient ──────────────────────────────

all_region_tables <- list()
all_event_tables <- list()
all_long_features <- list()
all_trajectories <- list()
all_progressions <- list()

for (pid in target_patients) {
  log_msg("\n", strrep("─", 50))
  log_msg("Processing patient: ", pid)
  log_msg(strrep("─", 50))

  patient_path <- file.path(PATHS$raw_data,
                             patient_index$path[patient_index$patient_id == pid])
  if (!dir.exists(patient_path)) {
    patient_path <- patient_index$path[patient_index$patient_id == pid]
  }

  # Load
  patient_data <- tryCatch(
    load_full_patient(patient_path),
    error = function(e) {
      log_msg("Failed to load: ", e$message, level = "ERROR")
      NULL
    }
  )
  if (is.null(patient_data)) next

  # Preprocess
  preprocessed <- tryCatch(
    preprocess_full_patient(patient_data, output_dir = PATHS$registered),
    error = function(e) {
      log_msg("Preprocessing failed: ", e$message, level = "ERROR")
      NULL
    }
  )
  if (is.null(preprocessed)) next

  # Extract regions
  region_data <- extract_all_patient_regions(preprocessed, pid)
  all_region_tables[[pid]] <- region_data$table

  # Track regions across timepoints
  if (length(region_data$regions_by_timepoint) >= 2) {
    tracking <- track_patient_regions(region_data$regions_by_timepoint)
    event_table <- create_event_table(tracking$all_events, pid)
    all_event_tables[[pid]] <- event_table

    # Compute longitudinal features
    long_feat <- compute_longitudinal_features(region_data$table, event_table)
    all_long_features[[pid]] <- long_feat

    # Analyze trajectories
    timepoints <- names(region_data$regions_by_timepoint)
    time_weeks <- seq(0, by = 8, length.out = length(timepoints))

    trajectories <- list()
    for (r in region_data$regions_by_timepoint[[1]]) {
      vols <- c()
      for (tp in timepoints) {
        tp_regions <- region_data$regions_by_timepoint[[tp]]
        matched <- Filter(function(x) x$region_type == r$region_type, tp_regions)
        vol <- if (length(matched) > 0) matched[[1]]$volume_cm3 else 0
        vols <- c(vols, vol)
      }
      traj <- analyze_region_trajectory(vols, time_weeks)
      trajectories[[r$region_id]] <- traj
    }
    all_trajectories[[pid]] <- trajectories

    # Classify progression
    progression <- classify_patient_progression(trajectories, event_table)
    all_progressions[[pid]] <- progression

    # Print report
    generate_patient_report(pid, region_data$table, event_table,
                            trajectories, progression)
  } else {
    log_msg("Only 1 timepoint — skipping tracking for ", pid)
  }
}

# ── Step 4: Aggregate Results ─────────────────────────────────

log_msg("\nAggregating results...")

combined_regions <- bind_rows(all_region_tables)
combined_events <- bind_rows(all_event_tables)
combined_long <- bind_rows(all_long_features)

# Save results
write_csv(combined_regions, file.path(PATHS$tables, "all_regions.csv"))
write_csv(combined_events, file.path(PATHS$tables, "all_events.csv"))
if (nrow(combined_long) > 0) {
  write_csv(combined_long, file.path(PATHS$tables, "longitudinal_features.csv"))
}

# Compute patient summaries
patient_summaries <- compute_patient_summary_features(
  combined_regions, combined_events, combined_long
)
write_csv(patient_summaries, file.path(PATHS$tables, "patient_summaries.csv"))

# ── Step 5: Clinical Validation (if RANO available) ───────────

rano_df <- NULL
for (csv_name in names(clinical_data)) {
  df <- clinical_data[[csv_name]]
  rano_cols <- grep("rano|RANO|progression|disease_state", names(df),
                    value = TRUE, ignore.case = TRUE)
  if (length(rano_cols) > 0) {
    rano_df <- df
    log_msg("Found RANO data in: ", csv_name)
    break
  }
}

if (!is.null(rano_df)) {
  log_msg("\nRunning clinical validation...")

  # Map framework predictions to RANO
  pred_df <- tibble(
    patient_id = names(all_progressions),
    prediction = sapply(all_progressions, function(p) p$classification)
  )

  # Run ablation study
  ablation <- run_ablation_study(patient_summaries, rano_df)
  print_validation_report(ablation)

  # Save results
  saveRDS(ablation, file.path(PATHS$results, "ablation_results.rds"))
  write_csv(format_ablation_table(ablation),
            file.path(PATHS$tables, "ablation_table.csv"))
} else {
  log_msg("No RANO data found — skipping clinical validation", level = "WARN")
  log_msg("Place RANO CSV in the LUMIERE data folder")
}

# ── Done ──────────────────────────────────────────────────────

cat("\n")
cat(strrep("=", 60), "\n")
cat("Pipeline Complete\n")
cat(strrep("=", 60), "\n")
cat("Patients processed: ", length(all_progressions), "\n")
cat("Regions extracted:  ", nrow(combined_regions), "\n")
cat("Events detected:    ", nrow(combined_events), "\n")
cat("Results saved to:   ", PATHS$results, "\n")
cat(strrep("=", 60), "\n")
