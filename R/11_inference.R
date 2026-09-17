# ──────────────────────────────────────────────────────────────
# Stage 11: Inference with Trained 3D U-Net
# Load trained model and segment new MRI scans
# ──────────────────────────────────────────────────────────────

library(torch)

if (!exists("UTILS_LOADED", envir = .GlobalEnv)) {
  SCRIPT_DIR <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) "R")
  source(file.path(SCRIPT_DIR, "utils.R"))
}

load_trained_unet <- function(model_path, device = "cpu") {
  is_jit <- grepl("_jit\\.pt$", model_path)
  jit_path <- if (is_jit) model_path else sub("\\.pt$", "_jit.pt", model_path)

  if (file.exists(jit_path)) {
    log_msg("Loading JIT model from: ", jit_path)
    model <- jit_load(jit_path)
    model <- model$to(device = device)
    log_msg("JIT model loaded successfully")
    return(model)
  }

  if (!file.exists(model_path)) {
    log_msg("Trained model not found: ", model_path, level = "WARN")
    return(NULL)
  }

  log_msg("Loading trained 3D U-Net from: ", model_path)
  model <- torch_load(model_path)
  model <- model$to(device = device)
  model$eval()
  log_msg("Model loaded successfully")
  model
}

segment_mri <- function(model, mri_stack, device = "cpu") {
  mri_tensor <- torch_tensor(mri_stack, dtype = torch_float())

  if (mri_tensor$dim() == 3) {
    mri_tensor <- mri_tensor$unsqueeze(1)$unsqueeze(1)
  } else if (mri_tensor$dim() == 4) {
    mri_tensor <- mri_tensor$permute(c(4, 1, 2, 3))$unsqueeze(1)
  }
  mri_tensor <- mri_tensor$to(device = device)

  with_no_grad({
    pred <- model(mri_tensor)
    seg <- pred$argmax(dim = 2)$squeeze(1)
  })

  as.array(seg$cpu())
}

segment_patient <- function(model, patient_preprocessed, device = "cpu") {
  results <- list()

  for (tp_name in names(patient_preprocessed)) {
    tp <- patient_preprocessed[[tp_name]]

    if (is.null(tp$stacked)) {
      log_msg("  ", tp_name, ": no stacked MRI, skipping", level = "WARN")
      next
    }

    log_msg("  Segmenting: ", tp_name)
    seg <- segment_mri(model, tp$stacked, device)

    results[[tp_name]] <- tp
    results[[tp_name]]$segmentation <- seg
    results[[tp_name]]$seg_source <- "trained_unet"

    labels <- sort(unique(as.integer(seg)))
    n_tumor <- sum(seg > 0)
    log_msg("    Labels: [", paste(labels, collapse = ","),
            "] | Tumor voxels: ", n_tumor)
  }

  results
}

run_full_inference_pipeline <- function(patient_data, preprocessed,
                                         model_path, device = "cpu") {
  model <- load_trained_unet(model_path, device)
  if (is.null(model)) return(NULL)

  segmented <- segment_patient(model, preprocessed, device)

  source("R/04_region_extractor.R")
  source("R/05_region_tracker.R")
  source("R/06_feature_extraction.R")
  source("R/07_progression_engine.R")

  pid <- patient_data$patient_id

  region_data <- extract_all_patient_regions(segmented, pid, min_vol = 50)

  if (length(region_data$regions_by_timepoint) < 2) {
    log_msg("Only 1 timepoint with regions — cannot track", level = "WARN")
    return(list(
      regions = region_data,
      events = NULL,
      trajectories = NULL,
      progression = NULL
    ))
  }

  tracking <- track_patient_regions(region_data$regions_by_timepoint)
  events <- create_event_table(tracking$all_events, pid)

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

  progression <- classify_patient_progression(trajectories, events)

  list(
    regions = region_data,
    events = events,
    trajectories = trajectories,
    progression = progression
  )
}
