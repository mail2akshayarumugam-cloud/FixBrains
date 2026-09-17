# ──────────────────────────────────────────────────────────────
# Stage 10: Export Training Data for 3D U-Net
# Converts preprocessed MRI + segmentation to .pt tensors
# Upload the output folder to Google Drive for Colab training
# ──────────────────────────────────────────────────────────────

library(tidyverse)
library(RNifti)
library(torch)

if (!exists("UTILS_LOADED", envir = .GlobalEnv)) {
  SCRIPT_DIR <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) "R")
  source(file.path(SCRIPT_DIR, "utils.R"))
}

export_training_data <- function(data_path, output_dir,
                                  target_size = c(64L, 64L, 64L),
                                  max_patients = NULL,
                                  project_root = ".") {
  source(file.path(project_root, "R/01_data_loader.R"))
  source(file.path(project_root, "R/02_preprocessing.R"))

  ensure_dir(output_dir)
  ensure_dir(file.path(output_dir, "mri"))
  ensure_dir(file.path(output_dir, "seg"))

  summary_data <- summarize_dataset(data_path)
  patients <- summary_data$patients

  if (!is.null(max_patients)) {
    patients <- patients %>%
      filter(n_timepoints >= 2) %>%
      head(max_patients)
  }

  manifest <- tibble(
    patient_id = character(),
    timepoint = character(),
    mri_file = character(),
    seg_file = character(),
    n_modalities = integer(),
    seg_labels = character(),
    tumor_voxels = integer()
  )

  total_exported <- 0L
  skipped <- 0L

  existing_manifest_path <- file.path(output_dir, "manifest.csv")
  if (file.exists(existing_manifest_path)) {
    existing <- read_csv(existing_manifest_path, show_col_types = FALSE)
    manifest <- existing
    total_exported <- nrow(existing)
    cat("Resuming: found", total_exported, "already exported samples\n")
    exported_keys <- paste0(existing$patient_id, "_", existing$timepoint)
  } else {
    exported_keys <- character(0)
  }

  for (i in seq_len(nrow(patients))) {
    pid <- patients$patient_id[i]
    cat("[", i, "/", nrow(patients), "]", pid, "\n")

    patient_data <- tryCatch(
      load_full_patient(patients$path[i]),
      error = function(e) { cat("  SKIP:", e$message, "\n"); NULL }
    )
    if (is.null(patient_data)) next

    for (tp_name in names(patient_data$timepoints)) {
      fname <- paste0(pid, "_", tp_name)

      if (fname %in% exported_keys) {
        skipped <- skipped + 1L
        cat("  ", tp_name, ": already exported, skipping\n")
        next
      }

      tp <- patient_data$timepoints[[tp_name]]

      if (is.null(tp$segmentation$mask)) {
        cat("  ", tp_name, ": no segmentation, skipping\n")
        next
      }

      seg_raw <- as.array(tp$segmentation$mask)
      roi_center <- find_roi_center(seg_raw)

      processed <- preprocess_patient_timepoint(
        tp$mri, target_size = target_size, roi_center = roi_center
      )

      if (length(processed) == 0) next

      stacked <- stack_modalities(processed)
      if (is.null(stacked)) next

      seg <- preprocess_segmentation(
        tp$segmentation$mask, target_size = target_size, roi_center = roi_center
      )
      if (is.null(seg)) next

      mri_tensor <- torch_tensor(stacked, dtype = torch_float16())
      mri_tensor <- mri_tensor$permute(c(4, 1, 2, 3))

      seg_tensor <- torch_tensor(seg, dtype = torch_uint8())

      mri_path <- file.path(output_dir, "mri", paste0(fname, ".pt"))
      seg_path <- file.path(output_dir, "seg", paste0(fname, ".pt"))

      torch_save(mri_tensor, mri_path)
      torch_save(seg_tensor, seg_path)

      labels <- sort(unique(as.integer(seg)))
      tumor_voxels <- sum(seg > 0)

      manifest <- bind_rows(manifest, tibble(
        patient_id = pid,
        timepoint = tp_name,
        mri_file = paste0("mri/", fname, ".pt"),
        seg_file = paste0("seg/", fname, ".pt"),
        n_modalities = dim(mri_tensor)[1],
        seg_labels = paste(labels, collapse = ","),
        tumor_voxels = tumor_voxels
      ))

      total_exported <- total_exported + 1L
      cat("  ", tp_name, ": exported (",
          paste(dim(mri_tensor), collapse = "x"), " | tumor:",
          tumor_voxels, "voxels)\n")

      # Save manifest incrementally so progress isn't lost on crash
      write_csv(manifest, file.path(output_dir, "manifest.csv"))
    }

    gc()
  }

  write_csv(manifest, file.path(output_dir, "manifest.csv"))
  cat("Skipped (already exported):", skipped, "\n")

  cat("\n========================================\n")
  cat("Export Complete\n")
  cat("========================================\n")
  cat("Total samples:", total_exported, "\n")
  cat("Output dir:   ", output_dir, "\n")
  cat("Manifest:     ", file.path(output_dir, "manifest.csv"), "\n")
  cat("MRI shape:    [4,", paste(target_size, collapse = ","), "]\n")
  cat("Seg shape:    [", paste(target_size, collapse = ","), "]\n")
  cat("\nUpload this folder to Google Drive for Colab training.\n")
  cat("========================================\n")

  manifest
}

if (sys.nframe() == 0) {
  setwd("C:/Users/mail2/OneDrive/Desktop/R project 1")
  manifest <- export_training_data(
    data_path = "data/raw/LUMIERE",
    output_dir = "data/training",
    target_size = c(48L, 48L, 48L),
    max_patients = NULL
  )
}
