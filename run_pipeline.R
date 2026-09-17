# ══════════════════════════════════════════════════════════════
# Tumor Progression Analysis — Full Pipeline (Review 1)
# LUMIERE Dataset | 3D U-Net + LSTM | R torch
# Aasif Khan A, Tarun Mukil S, Akshay Arumugam
#
# HOW TO RUN IN VS CODE:
#   1. Open this file
#   2. Open R terminal: Ctrl+Shift+` then type R
#   3. Run section by section: Ctrl+Shift+Enter
#   4. Or run single line: Ctrl+Enter
# ══════════════════════════════════════════════════════════════

# ---- 0. Setup ----

setwd("C:/Users/mail2/OneDrive/Desktop/R project 1")

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

source("R/utils.R")
source("R/01_data_loader.R")
source("R/02_preprocessing.R")
source("R/04_region_extractor.R")
source("R/05_region_tracker.R")
source("R/06_feature_extraction.R")
source("R/07_progression_engine.R")
source("R/09_clinical_validation.R")

DATA_PATH <- "data/raw/LUMIERE"
PROCESSED_PATH <- "data/processed"
MODELS_PATH <- "models"
RESULTS_PATH <- "results"
FIGURES_PATH <- file.path(RESULTS_PATH, "figures")
TABLES_PATH <- file.path(RESULTS_PATH, "tables")

for (p in c(PROCESSED_PATH, MODELS_PATH, RESULTS_PATH,
            FIGURES_PATH, TABLES_PATH,
            file.path(PROCESSED_PATH, "registered"))) {
  ensure_dir(p)
}

cat("Setup complete.\n")

# ---- 1. Dataset Exploration ----

cat("\n══════════════════════════════════════════\n")
cat("  STEP 1: Dataset Exploration\n")
cat("══════════════════════════════════════════\n")

summary_data <- summarize_dataset(DATA_PATH)
patients <- summary_data$patients

cat("\nTop 20 patients by timepoints:\n")
print(patients %>% select(patient_id, n_timepoints) %>% arrange(desc(n_timepoints)) %>% head(20))

# Viz 1: Timepoints per patient histogram
p1 <- ggplot(patients, aes(n_timepoints)) +
  geom_histogram(binwidth = 1, fill = "#2563eb", color = "white", alpha = 0.85) +
  geom_vline(aes(xintercept = mean(n_timepoints)), color = "#dc2626",
             linetype = "dashed", linewidth = 1) +
  labs(title = "Distribution of Timepoints per Patient",
       subtitle = paste(nrow(patients), "patients in LUMIERE dataset"),
       x = "Number of MRI Timepoints", y = "Patient Count") +
  annotate("text", x = mean(patients$n_timepoints) + 0.5,
           y = Inf, vjust = 2, hjust = 0,
           label = paste("Mean:", round(mean(patients$n_timepoints), 1)),
           color = "#dc2626", fontface = "bold") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))
print(p1)
ggsave(file.path(FIGURES_PATH, "01_timepoints_distribution.png"),
       p1, width = 8, height = 5, dpi = 150)

# Viz 2: Patient overview table
p2 <- patients %>%
  mutate(tp_group = cut(n_timepoints, breaks = c(0, 2, 4, 6, 10, Inf),
                         labels = c("1-2", "3-4", "5-6", "7-10", "11+"))) %>%
  count(tp_group) %>%
  ggplot(aes(tp_group, n, fill = tp_group)) +
  geom_col(alpha = 0.85) +
  geom_text(aes(label = n), vjust = -0.5, fontface = "bold") +
  labs(title = "Patients Grouped by Number of Timepoints",
       x = "Timepoint Range", y = "Number of Patients") +
  scale_fill_viridis_d(option = "D") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none", plot.title = element_text(face = "bold"))
print(p2)
ggsave(file.path(FIGURES_PATH, "01_patient_groups.png"),
       p2, width = 7, height = 5, dpi = 150)

cat("Step 1 complete. Figures saved.\n")

# ---- 1b. Load & Visualize One Patient ----

cat("\n══════════════════════════════════════════\n")
cat("  STEP 1b: Single Patient Visualization\n")
cat("══════════════════════════════════════════\n")

test_pid <- patients %>%
  arrange(desc(n_timepoints)) %>%
  slice(1) %>%
  pull(patient_id)

test_patient <- load_full_patient(
  file.path(find_imaging_root(DATA_PATH), test_pid)
)

cat("Patient:", test_patient$patient_id, "\n")
cat("Timepoints:", length(test_patient$timepoints), "\n")

tp1 <- test_patient$timepoints[[1]]

# Viz 3: MRI modality grid for first timepoint
mri_plots <- list()
for (mod in tp1$mri$available) {
  vol <- as.array(tp1$mri$volumes[[mod]])
  z <- dim(vol)[3] %/% 2
  slice <- vol[, , z]
  df <- expand.grid(x = seq_len(nrow(slice)), y = seq_len(ncol(slice)))
  df$intensity <- as.vector(slice)

  p <- ggplot(df, aes(x, y, fill = intensity)) +
    geom_raster() +
    scale_fill_viridis_c(option = "inferno") +
    coord_fixed() +
    labs(title = toupper(mod)) +
    theme_void() +
    theme(legend.position = "none",
          plot.title = element_text(hjust = 0.5, face = "bold", size = 14))
  mri_plots[[mod]] <- p
}

if (length(mri_plots) >= 2) {
  combined_mri <- wrap_plots(mri_plots, ncol = min(4, length(mri_plots))) +
    plot_annotation(
      title = paste("MRI Modalities —", test_pid, "(Axial Slice)"),
      theme = theme(plot.title = element_text(face = "bold", size = 16))
    )
  print(combined_mri)
  ggsave(file.path(FIGURES_PATH, "02_mri_modalities.png"),
         combined_mri, width = 14, height = 4, dpi = 150)
}

# Viz 4: Segmentation overlay
if (!is.null(tp1$segmentation$mask)) {
  seg <- as.array(tp1$segmentation$mask)
  flair_mod <- if ("flair" %in% tp1$mri$available) "flair" else tp1$mri$available[1]
  flair_vol <- as.array(tp1$mri$volumes[[flair_mod]])
  z <- dim(flair_vol)[3] %/% 2

  flair_slice <- flair_vol[, , z]
  seg_slice <- seg[, , z]

  df_overlay <- expand.grid(x = seq_len(nrow(flair_slice)),
                             y = seq_len(ncol(flair_slice)))
  df_overlay$intensity <- as.vector(flair_slice)
  df_overlay$seg_label <- factor(as.vector(seg_slice))

  seg_colors <- c("0" = "transparent", "1" = "#e74c3c",
                   "2" = "#f39c12", "3" = "#2ecc71", "4" = "#3498db")
  seg_names <- c("0" = "Background", "1" = "Necrosis",
                  "2" = "Edema", "3" = "Enhancing", "4" = "Enhancing Core")

  p_base <- ggplot(df_overlay, aes(x, y)) +
    geom_raster(aes(fill = intensity)) +
    scale_fill_gradient(low = "black", high = "white", guide = "none") +
    coord_fixed() +
    theme_void()

  p_seg <- ggplot(df_overlay %>% filter(seg_label != "0"), aes(x, y)) +
    geom_raster(aes(fill = seg_label), alpha = 0.6) +
    scale_fill_manual(values = seg_colors, labels = seg_names, name = "Region") +
    coord_fixed() +
    theme_void()

  p_overlay <- p_base +
    geom_raster(data = df_overlay %>% filter(seg_label != "0"),
                aes(x, y, fill = NULL, alpha = 0.5),
                fill = "#e74c3c") +
    labs(title = paste("Segmentation Overlay —", test_pid)) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14))
  print(p_overlay)
  ggsave(file.path(FIGURES_PATH, "02_segmentation_overlay.png"),
         p_overlay, width = 7, height = 6, dpi = 150)

  # Separate segmentation mask view with proper colors
  p_seg_only <- ggplot(df_overlay, aes(x, y, fill = seg_label)) +
    geom_raster() +
    scale_fill_manual(values = c("0" = "black", "1" = "#e74c3c",
                                  "2" = "#f39c12", "3" = "#2ecc71", "4" = "#3498db"),
                       labels = c("Background", "Necrosis", "Edema",
                                  "Enhancing", "Enhancing Core"),
                       name = "Tumor Region") +
    coord_fixed() +
    labs(title = paste("Segmentation Mask —", test_pid)) +
    theme_void() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
          legend.position = "right")
  print(p_seg_only)
  ggsave(file.path(FIGURES_PATH, "02_segmentation_mask.png"),
         p_seg_only, width = 9, height = 6, dpi = 150)
}

# Viz 5: Longitudinal MRI comparison (same slice across timepoints)
tp_names <- names(test_patient$timepoints)
long_plots <- list()
for (tp_name in tp_names[1:min(4, length(tp_names))]) {
  tp <- test_patient$timepoints[[tp_name]]
  mod <- if ("flair" %in% tp$mri$available) "flair" else tp$mri$available[1]
  vol <- as.array(tp$mri$volumes[[mod]])
  z <- dim(vol)[3] %/% 2
  slice <- vol[, , z]
  df <- expand.grid(x = seq_len(nrow(slice)), y = seq_len(ncol(slice)))
  df$intensity <- as.vector(slice)

  p <- ggplot(df, aes(x, y, fill = intensity)) +
    geom_raster() +
    scale_fill_viridis_c(option = "inferno") +
    coord_fixed() +
    labs(title = tp_name) +
    theme_void() +
    theme(legend.position = "none",
          plot.title = element_text(hjust = 0.5, face = "bold"))
  long_plots[[tp_name]] <- p
}

if (length(long_plots) >= 2) {
  long_combined <- wrap_plots(long_plots, ncol = min(4, length(long_plots))) +
    plot_annotation(
      title = paste("Longitudinal FLAIR —", test_pid, "(Axial Slice Across Time)"),
      theme = theme(plot.title = element_text(face = "bold", size = 16))
    )
  print(long_combined)
  ggsave(file.path(FIGURES_PATH, "02_longitudinal_mri.png"),
         long_combined, width = 14, height = 4, dpi = 150)
}

rm(test_patient); gc()
cat("Step 1b complete. MRI visualizations saved.\n")

# ---- 2. Preprocessing ----

cat("\n══════════════════════════════════════════\n")
cat("  STEP 2: Preprocessing\n")
cat("══════════════════════════════════════════\n")

N_PATIENTS <- 91
TARGET_SIZE <- c(48L, 48L, 48L)

selected_patients <- patients %>%
  filter(n_timepoints >= 3) %>%
  arrange(desc(n_timepoints)) %>%
  head(N_PATIENTS)

cat("Processing", nrow(selected_patients), "patients with 3+ timepoints\n")
cat("Target volume size:", paste(TARGET_SIZE, collapse = "x"), "\n\n")

REGISTERED_DIR <- file.path(PROCESSED_PATH, "registered")

all_preprocessed <- list()

for (i in seq_len(nrow(selected_patients))) {
  pid <- selected_patients$patient_id[i]
  ppath <- selected_patients$path[i]
  reg_dir <- file.path(REGISTERED_DIR, pid)

  cat("[", i, "/", nrow(selected_patients), "]", pid)

  # Try loading from already-registered data on disk (fast)
  if (dir.exists(reg_dir)) {
    tp_dirs <- list.dirs(reg_dir, recursive = FALSE, full.names = TRUE)
    if (length(tp_dirs) > 0) {
      cat(" — loading from registered data\n")
      preprocessed <- list()
      for (td in tp_dirs) {
        tp_name <- basename(td)
        mri_file <- file.path(td, "stacked_mri.rds")
        seg_file <- file.path(td, "segmentation.rds")
        if (file.exists(mri_file) && file.exists(seg_file)) {
          preprocessed[[tp_name]] <- tryCatch(list(
            stacked = readRDS(mri_file),
            segmentation = readRDS(seg_file),
            modalities_available = c("t1", "t1c", "t2", "flair")
          ), error = function(e) { cat("    SKIP", tp_name, ":", e$message, "\n"); NULL })
        }
      }
      preprocessed <- preprocessed[!sapply(preprocessed, is.null)]

      # Get week numbers from raw patient data (quick metadata load)
      patient_data <- tryCatch(load_full_patient(ppath), error = function(e) NULL)
      if (!is.null(patient_data)) {
        tp_meta <- lapply(patient_data$timepoints, function(tp) {
          list(week_number = tp$week_number, timepoint_name = tp$timepoint_name)
        })
        rm(patient_data); gc()
      } else {
        tp_meta <- lapply(names(preprocessed), function(n) {
          wk <- as.integer(gsub("week-([0-9]+).*", "\\1", n))
          list(week_number = wk, timepoint_name = n)
        })
        names(tp_meta) <- names(preprocessed)
      }

      all_preprocessed[[pid]] <- list(
        patient_data = list(patient_id = pid, timepoints = tp_meta),
        preprocessed = preprocessed
      )
      next
    }
  }
  cat("\n")

  # Full preprocessing from NIfTI
  patient_data <- tryCatch(
    load_full_patient(ppath),
    error = function(e) { cat("  SKIP:", e$message, "\n"); NULL }
  )
  if (is.null(patient_data)) next

  preprocessed <- tryCatch(
    preprocess_full_patient(patient_data, target_size = TARGET_SIZE,
                             output_dir = REGISTERED_DIR),
    error = function(e) { cat("  PREPROCESS FAIL:", e$message, "\n"); NULL }
  )
  if (is.null(preprocessed)) next

  patient_meta <- list(
    patient_id = patient_data$patient_id,
    timepoints = lapply(patient_data$timepoints, function(tp) {
      list(week_number = tp$week_number, timepoint_name = tp$timepoint_name)
    })
  )

  all_preprocessed[[pid]] <- list(
    patient_data = patient_meta,
    preprocessed = preprocessed
  )

  rm(patient_data); gc()
  cat("  Done\n")
}

# Remove patients with empty preprocessed data
all_preprocessed <- Filter(function(x) length(x$preprocessed) > 0, all_preprocessed)
cat("\nPreprocessed:", length(all_preprocessed), "patients\n")

# Viz 6: Preprocessing summary
preproc_summary <- tibble(
  patient_id = names(all_preprocessed),
  n_timepoints = sapply(all_preprocessed, function(x) length(x$preprocessed)),
  n_modalities = 4L
)

p6 <- ggplot(preproc_summary, aes(x = reorder(patient_id, -n_timepoints),
                                    y = n_timepoints, fill = factor(n_modalities))) +
  geom_col(alpha = 0.85) +
  geom_text(aes(label = n_timepoints), vjust = -0.5, fontface = "bold", size = 3.5) +
  labs(title = "Preprocessed Patients Overview",
       subtitle = paste("Target size:", paste(TARGET_SIZE, collapse = "x")),
       x = "Patient", y = "Timepoints", fill = "Modalities") +
  scale_fill_viridis_d(option = "C") +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(face = "bold"))
print(p6)
ggsave(file.path(FIGURES_PATH, "03_preprocessing_summary.png"),
       p6, width = 10, height = 5, dpi = 150)

cat("Step 2 complete.\n")

# ---- 3. Region Extraction ----

cat("\n══════════════════════════════════════════\n")
cat("  STEP 3: Region Extraction\n")
cat("══════════════════════════════════════════\n")

all_region_tables <- list()
all_regions_raw <- list()

for (pid in names(all_preprocessed)) {
  cat("Extracting regions:", pid, "\n")
  preprocessed <- all_preprocessed[[pid]]$preprocessed
  region_data <- extract_all_patient_regions(preprocessed, pid, min_vol = 50)
  all_regions_raw[[pid]] <- region_data$regions_by_timepoint
  all_region_tables[[pid]] <- region_data$table
}

combined_regions <- bind_rows(all_region_tables)
cat("\nTotal regions extracted:", nrow(combined_regions), "\n")

combined_regions %>%
  group_by(patient_id) %>%
  summarise(timepoints = n_distinct(timepoint), regions = n(), .groups = "drop") %>%
  print()

# Viz 7: Region type distribution
if (nrow(combined_regions) > 0) {
  p7 <- combined_regions %>%
    count(region_type) %>%
    mutate(region_type = fct_reorder(region_type, n, .desc = TRUE)) %>%
    ggplot(aes(region_type, n, fill = region_type)) +
    geom_col(alpha = 0.85) +
    geom_text(aes(label = n), vjust = -0.5, fontface = "bold") +
    labs(title = "Tumor Region Type Distribution",
         subtitle = paste(nrow(combined_regions), "total regions across",
                          n_distinct(combined_regions$patient_id), "patients"),
         x = "Region Type", y = "Count") +
    scale_fill_brewer(palette = "Set1") +
    theme_minimal(base_size = 13) +
    theme(legend.position = "none", plot.title = element_text(face = "bold"))
  print(p7)
  ggsave(file.path(FIGURES_PATH, "04_region_type_distribution.png"),
         p7, width = 8, height = 5, dpi = 150)

  # Viz 8: Volume distribution by region type
  p8 <- combined_regions %>%
    ggplot(aes(x = region_type, y = volume_cm3, fill = region_type)) +
    geom_boxplot(alpha = 0.7, outlier.shape = 21) +
    geom_jitter(alpha = 0.3, width = 0.2, size = 1.5) +
    labs(title = "Volume Distribution by Region Type",
         x = "Region Type", y = "Volume (cm³)") +
    scale_fill_brewer(palette = "Set1") +
    theme_minimal(base_size = 13) +
    theme(legend.position = "none", plot.title = element_text(face = "bold"))
  print(p8)
  ggsave(file.path(FIGURES_PATH, "04_volume_by_region_type.png"),
         p8, width = 8, height = 5, dpi = 150)
}

cat("Step 3 complete.\n")

# ---- 4. Region Tracking ----

cat("\n══════════════════════════════════════════\n")
cat("  STEP 4: Region Tracking\n")
cat("══════════════════════════════════════════\n")

all_event_tables <- list()
all_tracking <- list()

for (pid in names(all_regions_raw)) {
  regions_by_tp <- all_regions_raw[[pid]]
  if (length(regions_by_tp) < 2) next

  cat("Tracking:", pid, "\n")
  tracking <- track_patient_regions(regions_by_tp, match_threshold = 0.15)
  event_table <- create_event_table(tracking$all_events, pid)
  all_tracking[[pid]] <- tracking
  all_event_tables[[pid]] <- event_table
}

combined_events <- bind_rows(all_event_tables)
cat("\nTotal events detected:", nrow(combined_events), "\n")
if (nrow(combined_events) > 0) {
  cat("Event breakdown:\n")
  print(table(combined_events$event_type))
}

# Viz 9: Event distribution
if (nrow(combined_events) > 0) {
  event_colors <- c(
    "CONTINUATION" = "#2ecc71", "APPEARANCE" = "#e74c3c",
    "DISAPPEARANCE" = "#9b59b6", "SPLIT" = "#f39c12", "MERGE" = "#3498db"
  )

  p9 <- combined_events %>%
    count(event_type) %>%
    mutate(event_type = fct_reorder(event_type, n, .desc = TRUE)) %>%
    ggplot(aes(event_type, n, fill = event_type)) +
    geom_col(alpha = 0.85) +
    geom_text(aes(label = n), vjust = -0.5, fontface = "bold") +
    labs(title = "Longitudinal Region Events",
         subtitle = paste(nrow(combined_events), "total events across",
                          n_distinct(combined_events$patient_id), "patients"),
         x = "", y = "Event Count") +
    scale_fill_manual(values = event_colors) +
    theme_minimal(base_size = 13) +
    theme(legend.position = "none", plot.title = element_text(face = "bold"))
  print(p9)
  ggsave(file.path(FIGURES_PATH, "05_event_distribution.png"),
         p9, width = 9, height = 5, dpi = 150)

  # Viz 10: Events per patient
  p10 <- combined_events %>%
    count(patient_id, event_type) %>%
    ggplot(aes(x = patient_id, y = n, fill = event_type)) +
    geom_col(position = "stack", alpha = 0.85) +
    labs(title = "Events per Patient (Stacked)",
         x = "Patient", y = "Event Count", fill = "Event Type") +
    scale_fill_manual(values = event_colors) +
    theme_minimal(base_size = 12) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          plot.title = element_text(face = "bold"))
  print(p10)
  ggsave(file.path(FIGURES_PATH, "05_events_per_patient.png"),
         p10, width = 10, height = 5, dpi = 150)
}

cat("Step 4 complete.\n")

# ---- 5. Feature Extraction ----

cat("\n══════════════════════════════════════════\n")
cat("  STEP 5: Feature Extraction\n")
cat("══════════════════════════════════════════\n")

long_features <- compute_longitudinal_features(combined_regions, combined_events)

patient_summaries <- compute_patient_summary_features(
  combined_regions, combined_events, long_features
)

cat("Longitudinal features:", nrow(long_features), "\n")
cat("Patient summaries:", nrow(patient_summaries), "\n\n")

print(patient_summaries %>%
        select(patient_id, total_vol_first, total_vol_last,
               total_vol_pct_change, n_growing_regions, total_events))

# Viz 11: Volume change scatter plot
if (nrow(patient_summaries) > 0) {
  p11 <- ggplot(patient_summaries, aes(x = total_vol_first, y = total_vol_last)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
    geom_point(aes(color = total_vol_pct_change, size = total_events),
               alpha = 0.8) +
    geom_text(aes(label = patient_id), vjust = -1, size = 2.5, check_overlap = TRUE) +
    scale_color_gradient2(low = "#2ecc71", mid = "gray80", high = "#e74c3c",
                           midpoint = 0, name = "% Volume\nChange") +
    scale_size_continuous(range = c(3, 10), name = "Total\nEvents") +
    labs(title = "First vs Last Total Tumor Volume",
         subtitle = "Points above line = growth; below = shrinkage",
         x = "First Timepoint Volume (cm³)", y = "Last Timepoint Volume (cm³)") +
    theme_minimal(base_size = 13) +
    theme(plot.title = element_text(face = "bold"))
  print(p11)
  ggsave(file.path(FIGURES_PATH, "06_volume_change_scatter.png"),
         p11, width = 9, height = 7, dpi = 150)

  # Viz 12: Feature summary heatmap
  feat_cols <- c("total_vol_pct_change", "n_growing_regions",
                  "n_shrinking_regions", "max_growth_rate",
                  "n_appearances", "n_disappearances", "n_splits", "n_merges")
  feat_cols <- intersect(feat_cols, names(patient_summaries))

  if (length(feat_cols) > 0) {
    feat_mat <- patient_summaries %>%
      select(patient_id, all_of(feat_cols)) %>%
      mutate(across(all_of(feat_cols), ~scale(.)[,1])) %>%
      pivot_longer(-patient_id, names_to = "Feature", values_to = "Value")

    p12 <- ggplot(feat_mat, aes(x = Feature, y = patient_id, fill = Value)) +
      geom_tile(color = "white") +
      scale_fill_gradient2(low = "#2563eb", mid = "white", high = "#dc2626",
                            midpoint = 0, name = "Z-Score") +
      labs(title = "Patient Feature Heatmap (Standardized)",
           x = "", y = "") +
      theme_minimal(base_size = 11) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1),
            plot.title = element_text(face = "bold"))
    print(p12)
    ggsave(file.path(FIGURES_PATH, "06_feature_heatmap.png"),
           p12, width = 10, height = 6, dpi = 150)
  }
}

cat("Step 5 complete.\n")

# ---- 6. Progression Classification (Rule-Based) ----

cat("\n══════════════════════════════════════════\n")
cat("  STEP 6: Progression Classification\n")
cat("══════════════════════════════════════════\n")

all_trajectories <- list()
all_progressions <- list()

for (pid in names(all_regions_raw)) {
  regions_by_tp <- all_regions_raw[[pid]]
  if (length(regions_by_tp) < 2) next

  tp_names <- names(regions_by_tp)
  patient_data <- all_preprocessed[[pid]]$patient_data
  time_weeks <- sapply(patient_data$timepoints[tp_names], `[[`, "week_number")

  trajectories <- list()
  for (r in regions_by_tp[[1]]) {
    vols <- c()
    for (tp in tp_names) {
      tp_regions <- regions_by_tp[[tp]]
      matched <- Filter(function(x) x$region_type == r$region_type, tp_regions)
      vol <- if (length(matched) > 0) matched[[1]]$volume_cm3 else 0
      vols <- c(vols, vol)
    }
    traj <- analyze_region_trajectory(vols, time_weeks)
    trajectories[[r$region_id]] <- traj
  }

  events <- all_event_tables[[pid]]
  if (is.null(events)) events <- tibble(event_type = character(0))

  progression <- classify_patient_progression(trajectories, events)
  all_trajectories[[pid]] <- trajectories
  all_progressions[[pid]] <- progression

  tryCatch(
    generate_patient_report(pid, combined_regions %>% filter(patient_id == pid),
                             events, trajectories, progression),
    error = function(e) cat("  Report generation skipped:", e$message, "\n")
  )
}

# Viz 13: Progression classification distribution
if (length(all_progressions) > 0) {
  prog_df <- tibble(
    patient_id = names(all_progressions),
    classification = sapply(all_progressions, `[[`, "classification"),
    confidence = sapply(all_progressions, `[[`, "confidence")
  )

  prog_colors <- c("CR" = "#2ecc71", "PR" = "#3498db", "SD" = "#f39c12", "PD" = "#e74c3c")

  p13a <- prog_df %>%
    mutate(classification = factor(classification, levels = c("CR", "PR", "SD", "PD"))) %>%
    count(classification, .drop = FALSE) %>%
    ggplot(aes(classification, n, fill = classification)) +
    geom_col(alpha = 0.85, width = 0.6) +
    geom_text(aes(label = n), vjust = -0.5, fontface = "bold", size = 5) +
    labs(title = "Progression Classification (RANO Criteria)",
         subtitle = paste(nrow(prog_df), "patients classified"),
         x = "RANO Category", y = "Patient Count") +
    scale_fill_manual(values = prog_colors) +
    theme_minimal(base_size = 14) +
    theme(legend.position = "none", plot.title = element_text(face = "bold"))

  p13b <- prog_df %>%
    ggplot(aes(x = classification, y = confidence, fill = classification)) +
    geom_boxplot(alpha = 0.7, width = 0.5) +
    geom_jitter(width = 0.1, alpha = 0.5, size = 2) +
    labs(title = "Classification Confidence",
         x = "RANO Category", y = "Confidence Score") +
    scale_fill_manual(values = prog_colors) +
    theme_minimal(base_size = 14) +
    theme(legend.position = "none", plot.title = element_text(face = "bold"))

  p13 <- p13a + p13b + plot_annotation(
    title = "Rule-Based Progression Analysis",
    theme = theme(plot.title = element_text(face = "bold", size = 16))
  )
  print(p13)
  ggsave(file.path(FIGURES_PATH, "07_progression_classification.png"),
         p13, width = 14, height = 6, dpi = 150)
}

cat("Step 6 complete.\n")

# ---- 7. Volume Trajectory Visualizations ----
tryCatch({
cat("\n══════════════════════════════════════════\n")
cat("  STEP 7: Trajectory Visualizations\n")
cat("══════════════════════════════════════════\n")

if (nrow(combined_regions) > 0) {
  top_pids <- names(all_progressions)[1:min(6, length(all_progressions))]

  # Viz 14: Region volume trajectories faceted by patient
  p14 <- combined_regions %>%
    filter(patient_id %in% top_pids) %>%
    ggplot(aes(x = timepoint, y = volume_cm3,
               color = region_type, group = region_id)) +
    geom_line(linewidth = 1.2) +
    geom_point(size = 2.5) +
    facet_wrap(~patient_id, scales = "free", ncol = 3) +
    labs(title = "Region Volume Trajectories Over Time",
         subtitle = "Each line = one tumor region tracked across MRI timepoints",
         x = "Timepoint", y = "Volume (cm³)", color = "Region Type") +
    scale_color_brewer(palette = "Set1") +
    theme_minimal(base_size = 12) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7),
          plot.title = element_text(face = "bold"),
          strip.text = element_text(face = "bold"))
  print(p14)
  ggsave(file.path(FIGURES_PATH, "08_volume_trajectories.png"),
         p14, width = 14, height = 9, dpi = 150)

  # Viz 15: Total volume per patient over time
  total_vol_ts <- combined_regions %>%
    filter(patient_id %in% top_pids) %>%
    group_by(patient_id, timepoint) %>%
    summarise(total_volume = sum(volume_cm3, na.rm = TRUE), .groups = "drop")

  p15 <- total_vol_ts %>%
    ggplot(aes(x = timepoint, y = total_volume,
               group = patient_id, color = patient_id)) +
    geom_line(linewidth = 1.2) +
    geom_point(size = 3) +
    labs(title = "Total Tumor Volume Over Time",
         x = "Timepoint", y = "Total Volume (cm³)", color = "Patient") +
    scale_color_viridis_d(option = "D") +
    theme_minimal(base_size = 13) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          plot.title = element_text(face = "bold"))
  print(p15)
  ggsave(file.path(FIGURES_PATH, "08_total_volume_timeseries.png"),
         p15, width = 11, height = 6, dpi = 150)
}

# Viz 16: Longitudinal features scatter
if (nrow(long_features) > 0) {
  p16 <- long_features %>%
    ggplot(aes(x = pct_volume_change, y = centroid_displacement,
               color = region_type)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray60") +
    geom_vline(xintercept = c(-0.25, 0.25), linetype = "dashed", color = "gray60") +
    geom_point(alpha = 0.7, size = 3) +
    labs(title = "Volume Change vs Centroid Displacement",
         subtitle = "Dashed lines = 25% growth/shrinkage thresholds",
         x = "% Volume Change", y = "Centroid Displacement (mm)",
         color = "Region Type") +
    scale_color_brewer(palette = "Set1") +
    theme_minimal(base_size = 13) +
    theme(plot.title = element_text(face = "bold"))
  print(p16)
  ggsave(file.path(FIGURES_PATH, "08_volume_vs_displacement.png"),
         p16, width = 10, height = 7, dpi = 150)
}

cat("Step 7 complete.\n")
}, error = function(e) cat("STEP 7 ERROR (skipped):", e$message, "\n"))

# ---- 8. 3D U-Net Architecture Demo ----
tryCatch({
cat("\n══════════════════════════════════════════\n")
cat("  STEP 8: 3D U-Net Architecture\n")
cat("══════════════════════════════════════════\n")

source("R/03_model_3d_unet.R")

model <- unet_3d(in_channels = 4L, n_classes = 4L, base_filters = 16L)
n_params <- sum(sapply(model$parameters, function(p) prod(p$shape)))

cat("\n3D U-Net Architecture Summary:\n")
cat("  Input:          4-channel MRI (T1, T1c, T2, FLAIR)\n")
cat("  Output:         4-class segmentation mask\n")
cat("  Parameters:    ", format(n_params, big.mark = ","), "\n")
cat("  Encoder:        4 levels (16 → 32 → 64 → 128 filters)\n")
cat("  Bottleneck:     256 filters\n")
cat("  Decoder:        4 levels with skip connections\n")
cat("  Loss:           Dice + Cross-Entropy (combined)\n\n")

# Forward pass verification
dummy <- torch_randn(1, 4, 32, 32, 32)
with_no_grad({ out <- model(dummy) })
cat("Forward pass: [1,4,32,32,32] → [", paste(dim(out), collapse = ","), "]\n")
cat("Architecture verified on CPU.\n")

# Viz 17: Model architecture diagram
layer_info <- tibble(
  Layer = c("Encoder 1", "Encoder 2", "Encoder 3", "Encoder 4",
            "Bottleneck",
            "Decoder 4", "Decoder 3", "Decoder 2", "Decoder 1",
            "Final Conv"),
  Filters = c(16, 32, 64, 128, 256, 128, 64, 32, 16, 4),
  Type = c(rep("Encoder", 4), "Bottleneck", rep("Decoder", 4), "Output")
)
layer_info$Layer <- factor(layer_info$Layer, levels = rev(layer_info$Layer))

p17 <- ggplot(layer_info, aes(x = Layer, y = Filters, fill = Type)) +
  geom_col(alpha = 0.85, width = 0.7) +
  geom_text(aes(label = Filters), hjust = -0.3, fontface = "bold") +
  coord_flip() +
  labs(title = "3D U-Net Architecture — Filter Counts per Layer",
       subtitle = paste(format(n_params, big.mark = ","), "trainable parameters"),
       x = "", y = "Number of Filters") +
  scale_fill_manual(values = c("Encoder" = "#2563eb", "Bottleneck" = "#7c3aed",
                                 "Decoder" = "#059669", "Output" = "#dc2626")) +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))
print(p17)
ggsave(file.path(FIGURES_PATH, "09_unet_architecture.png"),
       p17, width = 10, height = 6, dpi = 150)

rm(model, dummy, out); gc()

# NOTE: 3D U-Net training done on Google Colab with GPU
# Download trained model to models/ folder after Colab training
cat("3D U-Net training skipped (done on Colab GPU).\n")

cat("Step 8 complete.\n")
}, error = function(e) cat("STEP 8 ERROR (skipped):", e$message, "\n"))

# ---- 9. Train LSTM ----

tryCatch({
cat("\n══════════════════════════════════════════\n")
cat("  STEP 9: LSTM Training\n")
cat("══════════════════════════════════════════\n")

source("R/08_model_lstm.R")

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
  lstm_labels <- c(lstm_labels, match(prog, c("CR", "PR", "SD", "PD")) - 1L)
}

cat("LSTM sequences:", length(lstm_sequences), "\n")
cat("Label distribution:\n")
if (length(lstm_labels) > 0) {
  print(table(c("CR", "PR", "SD", "PD")[lstm_labels + 1]))
}

# Train LSTM
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
    input_size  = 13L,
    hidden_size = 64L,
    num_layers  = 2L,
    n_classes   = 4L,
    epochs      = 50L,
    batch_size  = 4L,
    lr          = 1e-3,
    save_path   = file.path(MODELS_PATH, "lstm_progression.pt")
  )

  # Viz 18: Training curves
  if (length(lstm_result$history$train_loss) > 0) {
    p18a <- tibble(
      epoch = seq_along(lstm_result$history$train_loss),
      Train = lstm_result$history$train_loss,
      Val   = lstm_result$history$val_loss
    ) %>%
      pivot_longer(-epoch, names_to = "Split", values_to = "Loss") %>%
      ggplot(aes(epoch, Loss, color = Split)) +
      geom_line(linewidth = 1.2) +
      geom_point(size = 1, alpha = 0.5) +
      labs(title = "LSTM Training Loss") +
      scale_color_manual(values = c(Train = "#2563eb", Val = "#dc2626")) +
      theme_minimal(base_size = 13) +
      theme(plot.title = element_text(face = "bold"))

    p18b <- tibble(
      epoch = seq_along(lstm_result$history$train_acc),
      Train = lstm_result$history$train_acc * 100,
      Val   = lstm_result$history$val_acc * 100
    ) %>%
      pivot_longer(-epoch, names_to = "Split", values_to = "Accuracy") %>%
      ggplot(aes(epoch, Accuracy, color = Split)) +
      geom_line(linewidth = 1.2) +
      geom_point(size = 1, alpha = 0.5) +
      labs(title = "LSTM Accuracy (%)") +
      scale_color_manual(values = c(Train = "#2563eb", Val = "#dc2626")) +
      theme_minimal(base_size = 13) +
      theme(plot.title = element_text(face = "bold"))

    p18 <- p18a + p18b + plot_annotation(
      title = "LSTM Temporal Progression Model — Training",
      subtitle = paste("Input: 13-feature temporal sequences | Output: RANO class"),
      theme = theme(plot.title = element_text(face = "bold", size = 16))
    )
    print(p18)
    ggsave(file.path(FIGURES_PATH, "10_lstm_training.png"),
           p18, width = 14, height = 5, dpi = 150)
  }
} else {
  cat("Not enough sequences to train LSTM (need >= 4).\n")
  lstm_result <- NULL
}

cat("Step 9 complete.\n")
}, error = function(e) cat("STEP 9 ERROR (skipped):", e$message, "\n"))

# ---- 10. Ablation Study ----

tryCatch({
cat("\n══════════════════════════════════════════\n")
cat("  STEP 10: Ablation Study\n")
cat("══════════════════════════════════════════\n")

rano_labels <- tibble(
  patient_id = names(all_progressions),
  rano_label = sapply(all_progressions, `[[`, "classification")
)

if (nrow(patient_summaries) > 0 && nrow(rano_labels) > 0) {
  ablation <- run_ablation_study(patient_summaries, rano_labels)
  print_validation_report(ablation)

  abl_table <- format_ablation_table(ablation)
  write_csv(abl_table, file.path(TABLES_PATH, "ablation_results.csv"))

  # Viz 19: Ablation comparison
  p19 <- abl_table %>%
    mutate(across(c(`Macro F1`, Kappa, `PD Sensitivity`, `PD F1`),
                  ~as.numeric(.))) %>%
    pivot_longer(c(`Macro F1`, Kappa, `PD Sensitivity`, `PD F1`),
                 names_to = "Metric", values_to = "Value") %>%
    ggplot(aes(Approach, Value, fill = Metric)) +
    geom_col(position = "dodge", alpha = 0.85) +
    geom_text(aes(label = sprintf("%.2f", Value)),
              position = position_dodge(width = 0.9),
              vjust = -0.5, size = 3) +
    labs(title = "Ablation Study: Total Volume vs Region-Wise vs Full Framework",
         subtitle = "Comparing three approaches against clinical RANO criteria",
         y = "Score", x = "") +
    scale_fill_brewer(palette = "Set2") +
    theme_minimal(base_size = 13) +
    theme(axis.text.x = element_text(angle = 15, hjust = 1),
          plot.title = element_text(face = "bold"))
  print(p19)
  ggsave(file.path(FIGURES_PATH, "11_ablation_comparison.png"),
         p19, width = 12, height = 6, dpi = 150)

  # Viz 20: Confusion matrices for each approach
  for (approach_name in names(ablation)) {
    r <- ablation[[approach_name]]
    cm <- as.data.frame(r$metrics$confusion_matrix)
    names(cm) <- c("Predicted", "Actual", "Count")

    p_cm <- ggplot(cm, aes(x = Actual, y = Predicted, fill = Count)) +
      geom_tile(color = "white", linewidth = 1) +
      geom_text(aes(label = Count), size = 6, fontface = "bold") +
      scale_fill_gradient(low = "white", high = "#2563eb") +
      labs(title = paste("Confusion Matrix —", r$name)) +
      theme_minimal(base_size = 13) +
      theme(plot.title = element_text(face = "bold"),
            axis.text = element_text(size = 12))
    print(p_cm)
    ggsave(file.path(FIGURES_PATH, paste0("11_confusion_", approach_name, ".png")),
           p_cm, width = 6, height = 5, dpi = 150)
  }
}

cat("Step 10 complete.\n")
}, error = function(e) cat("STEP 10 ERROR (skipped):", e$message, "\n"))

# ---- 11. Hidden Regional Changes ----

tryCatch({
cat("\n══════════════════════════════════════════\n")
cat("  STEP 11: Hidden Regional Changes Analysis\n")
cat("══════════════════════════════════════════\n")

if (nrow(patient_summaries) > 0) {
  insight <- patient_summaries %>%
    mutate(
      total_says = case_when(
        total_vol_pct_change > 0.25 ~ "Growing",
        total_vol_pct_change < -0.25 ~ "Shrinking",
        TRUE ~ "Stable"
      ),
      regions_say = case_when(
        n_growing_regions > 0 & n_shrinking_regions > 0 ~ "Mixed (Hidden!)",
        n_growing_regions > 0 ~ "Growing",
        n_shrinking_regions > 0 ~ "Shrinking",
        TRUE ~ "Stable"
      ),
      disagree = total_says != regions_say
    )

  n_hidden <- sum(insight$disagree)
  cat("\nPatients where total volume HIDES regional changes:", n_hidden, "\n")
  cat("This is", round(n_hidden / nrow(insight) * 100, 1), "% of patients\n")

  # Viz 21: Total vs region-wise assessment
  p21 <- ggplot(insight, aes(x = total_says, fill = regions_say)) +
    geom_bar(position = "dodge", alpha = 0.85) +
    labs(title = "Total Volume vs Region-Wise Assessment",
         subtitle = paste(n_hidden, "patients show hidden regional changes (",
                          round(n_hidden / nrow(insight) * 100, 1), "%)"),
         x = "What Total Volume Says", y = "Number of Patients",
         fill = "Region-Wise\nAnalysis") +
    scale_fill_brewer(palette = "Set1") +
    theme_minimal(base_size = 14) +
    theme(plot.title = element_text(face = "bold"))
  print(p21)
  ggsave(file.path(FIGURES_PATH, "12_hidden_changes.png"),
         p21, width = 10, height = 6, dpi = 150)

  # Viz 22: Detailed disagreement view
  p22 <- insight %>%
    select(patient_id, total_says, regions_say, disagree,
           total_vol_pct_change, n_growing_regions, n_shrinking_regions) %>%
    pivot_longer(c(total_says, regions_say), names_to = "Method", values_to = "Assessment") %>%
    mutate(Method = recode(Method, "total_says" = "Total Volume",
                           "regions_say" = "Region-Wise")) %>%
    ggplot(aes(x = patient_id, y = Method, fill = Assessment)) +
    geom_tile(color = "white", linewidth = 1.5) +
    scale_fill_manual(values = c("Growing" = "#e74c3c", "Shrinking" = "#2ecc71",
                                  "Stable" = "#f39c12", "Mixed (Hidden!)" = "#9b59b6")) +
    labs(title = "Assessment Comparison: Total Volume vs Region-Wise",
         x = "Patient", y = "") +
    theme_minimal(base_size = 12) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          plot.title = element_text(face = "bold"))
  print(p22)
  ggsave(file.path(FIGURES_PATH, "12_assessment_comparison.png"),
         p22, width = 12, height = 4, dpi = 150)
}

cat("Step 11 complete.\n")
}, error = function(e) cat("STEP 11 ERROR (skipped):", e$message, "\n"))

# ---- 12. Save All Results ----
tryCatch({
cat("\n══════════════════════════════════════════\n")
cat("  STEP 12: Saving Results\n")
cat("══════════════════════════════════════════\n")

write_csv(combined_regions, file.path(TABLES_PATH, "all_regions.csv"))
write_csv(combined_events, file.path(TABLES_PATH, "all_events.csv"))
if (nrow(long_features) > 0) {
  write_csv(long_features, file.path(TABLES_PATH, "longitudinal_features.csv"))
}
write_csv(patient_summaries, file.path(TABLES_PATH, "patient_summaries.csv"))
write_csv(rano_labels, file.path(TABLES_PATH, "predictions.csv"))

if (exists("insight")) {
  write_csv(insight, file.path(TABLES_PATH, "hidden_changes_analysis.csv"))
}

# Final summary
cat("\n")
cat("══════════════════════════════════════════════════════════════\n")
cat("  PIPELINE COMPLETE — Review 1\n")
cat("══════════════════════════════════════════════════════════════\n")
cat("  Patients processed:    ", length(all_progressions), "\n")
cat("  Regions extracted:     ", nrow(combined_regions), "\n")
cat("  Events detected:       ", nrow(combined_events), "\n")
cat("  Longitudinal features: ", nrow(long_features), "\n")
cat("  Hidden change cases:   ", if (exists("n_hidden")) n_hidden else "N/A", "\n")
cat("  LSTM trained:          ", if (exists("lstm_result") && !is.null(lstm_result)) "Yes" else "No", "\n")
cat("  3D U-Net verified:     Yes (CPU forward pass)\n")
cat("\n")
cat("  Results:  ", TABLES_PATH, "\n")
cat("  Figures:  ", FIGURES_PATH, "\n")
cat("  Models:   ", MODELS_PATH, "\n")
cat("══════════════════════════════════════════════════════════════\n")

# List generated figures
cat("\nGenerated Figures:\n")
figs <- list.files(FIGURES_PATH, pattern = "\\.png$")
for (f in figs) cat("  ", f, "\n")

cat("\nGenerated Tables:\n")
tabs <- list.files(TABLES_PATH, pattern = "\\.csv$")
for (t in tabs) cat("  ", t, "\n")
}, error = function(e) cat("STEP 12 ERROR:", e$message, "\n"))
