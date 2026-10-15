# ──────────────────────────────────────────────────────────────
# Stage 11: Inference with Trained 3D U-Net
# Load trained model and segment new MRI scans
# ──────────────────────────────────────────────────────────────

library(torch)

if (!exists("UTILS_LOADED", envir = .GlobalEnv)) {
  SCRIPT_DIR <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) "R")
  source(file.path(SCRIPT_DIR, "utils.R"))
}

if (!exists("label_connected_components_3d", mode = "function")) {
  SCRIPT_DIR <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) "R")
  source(file.path(SCRIPT_DIR, "04_region_extractor.R"))
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

postprocess_segmentation <- function(seg_array, min_blob_voxels = 50L,
                                     keep_largest_only = FALSE) {
  dims <- dim(seg_array)
  cleaned <- seg_array

  for (lbl in unique(as.integer(seg_array))) {
    if (lbl == 0L) next
    binary <- array(0L, dim = dims)
    binary[seg_array == lbl] <- 1L

    cc <- label_connected_components_3d(binary)

    if (keep_largest_only && cc$n_components > 1) {
      comp_sizes <- sapply(seq_len(cc$n_components),
                           function(c) sum(cc$labels == c))
      largest <- which.max(comp_sizes)
      for (comp in seq_len(cc$n_components)) {
        if (comp != largest) {
          cleaned[cc$labels == comp & seg_array == lbl] <- 0L
        }
      }
    } else {
      for (comp in seq_len(cc$n_components)) {
        comp_size <- sum(cc$labels == comp)
        if (comp_size < min_blob_voxels) {
          cleaned[cc$labels == comp & seg_array == lbl] <- 0L
        }
      }
    }
  }

  # Fill small holes inside tumor regions
  tumor_mask <- cleaned > 0
  if (sum(tumor_mask) > 0 && requireNamespace("mmand", quietly = TRUE)) {
    kernel <- mmand::shapeKernel(c(3, 3, 3), type = "diamond")
    closed <- mmand::dilate(mmand::erode(tumor_mask, kernel), kernel)
    holes_filled <- which(closed & !tumor_mask, arr.ind = TRUE)
    if (nrow(holes_filled) > 0) {
      for (i in seq_len(nrow(holes_filled))) {
        h <- holes_filled[i, ]
        neighbors <- c()
        for (d in list(c(-1,0,0),c(1,0,0),c(0,-1,0),c(0,1,0),c(0,0,-1),c(0,0,1))) {
          ni <- h + d
          if (all(ni >= 1) && all(ni <= dims)) {
            v <- cleaned[ni[1], ni[2], ni[3]]
            if (v > 0) neighbors <- c(neighbors, v)
          }
        }
        if (length(neighbors) > 0) {
          cleaned[h[1], h[2], h[3]] <- as.integer(names(which.max(table(neighbors))))
        }
      }
      log_msg("    Filled ", nrow(holes_filled), " hole voxels")
    }
  }

  n_removed <- sum(seg_array > 0) - sum(cleaned > 0)
  if (n_removed > 0) {
    log_msg("    Post-processing: removed ", n_removed, " noisy voxels")
  }
  cleaned
}

predict_probs <- function(model, mri_tensor, device = "cpu") {
  mri_tensor <- mri_tensor$to(device = device)
  with_no_grad({
    pred <- model(mri_tensor)
    probs <- nnf_softmax(pred, dim = 2)
  })
  probs
}

segment_mri_tta <- function(model, mri_stack, device = "cpu", n_tta = 8L) {
  mri_tensor <- torch_tensor(mri_stack, dtype = torch_float())
  if (mri_tensor$dim() == 3) {
    mri_tensor <- mri_tensor$unsqueeze(1)$unsqueeze(1)
  } else if (mri_tensor$dim() == 4) {
    mri_tensor <- mri_tensor$permute(c(4, 1, 2, 3))$unsqueeze(1)
  }

  avg_probs <- predict_probs(model, mri_tensor, device)

  flip_axes <- list(c(3), c(4), c(5), c(3,4), c(3,5), c(4,5), c(3,4,5))
  n_augments <- min(n_tta - 1L, length(flip_axes))

  if (n_augments > 0) {
    for (i in seq_len(n_augments)) {
      flipped <- torch_flip(mri_tensor, flip_axes[[i]])
      probs <- predict_probs(model, flipped, device)
      unflipped <- torch_flip(probs, flip_axes[[i]])
      avg_probs <- avg_probs + unflipped
    }
    avg_probs <- avg_probs / (n_augments + 1L)
  }

  seg <- avg_probs$argmax(dim = 2)$squeeze(1)
  as.array(seg$cpu())
}

segment_mri <- function(model, mri_stack, device = "cpu",
                        postprocess = TRUE, use_tta = TRUE) {
  if (use_tta) {
    log_msg("    Using test-time augmentation (8 passes)")
    seg_arr <- segment_mri_tta(model, mri_stack, device, n_tta = 8L)
  } else {
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
    seg_arr <- as.array(seg$cpu())
  }

  if (postprocess) {
    seg_arr <- postprocess_segmentation(seg_arr)
  }

  seg_arr
}

segment_patient <- function(model, patient_preprocessed, device = "cpu",
                            prefer_ground_truth = TRUE) {
  results <- list()

  for (tp_name in names(patient_preprocessed)) {
    tp <- patient_preprocessed[[tp_name]]

    if (is.null(tp$stacked)) {
      log_msg("  ", tp_name, ": no stacked MRI, skipping", level = "WARN")
      next
    }

    has_gt <- prefer_ground_truth && !is.null(tp$segmentation) &&
      sum(tp$segmentation > 0) > 0

    if (has_gt) {
      log_msg("  ", tp_name, ": using ground truth segmentation")
      seg <- tp$segmentation
      seg_source <- "ground_truth"
    } else {
      log_msg("  Segmenting: ", tp_name, " with U-Net + TTA")
      seg <- segment_mri(model, tp$stacked, device)
      seg_source <- "unet_v2_tta"
    }

    results[[tp_name]] <- tp
    results[[tp_name]]$segmentation <- seg
    results[[tp_name]]$seg_source <- seg_source

    labels <- sort(unique(as.integer(seg)))
    n_tumor <- sum(seg > 0)
    log_msg("    Labels: [", paste(labels, collapse = ","),
            "] | Tumor voxels: ", n_tumor)
  }

  results
}

run_full_inference_pipeline <- function(patient_data, preprocessed,
                                         model_path, device = "cpu",
                                         preloaded_model = NULL) {
  model <- if (!is.null(preloaded_model)) preloaded_model else load_trained_unet(model_path, device)
  if (is.null(model)) return(NULL)

  segmented <- segment_patient(model, preprocessed, device)

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
