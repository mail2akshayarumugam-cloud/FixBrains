# ──────────────────────────────────────────────────────────────
# Stage 2: MRI Preprocessing
# Normalize, resample, register longitudinal scans
# ──────────────────────────────────────────────────────────────

library(RNifti)
library(mmand)

if (!exists("UTILS_LOADED", envir = .GlobalEnv)) {
  SCRIPT_DIR <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) "R")
  source(file.path(SCRIPT_DIR, "utils.R"))
}

zscore_normalize <- function(volume, brain_mask = NULL) {
  if (!is.null(brain_mask)) {
    brain_voxels <- volume[brain_mask > 0]
  } else {
    brain_voxels <- volume[volume > quantile(volume, 0.02)]
  }

  if (length(brain_voxels) == 0 || sd(brain_voxels) < 1e-8) return(volume)

  mu <- mean(brain_voxels)
  sigma <- sd(brain_voxels)
  normalized <- (volume - mu) / sigma

  if (!is.null(brain_mask)) normalized[brain_mask == 0] <- 0
  normalized
}

create_brain_mask <- function(volume, threshold_quantile = 0.05) {
  pos_voxels <- volume[volume > 0]
  if (length(pos_voxels) == 0) return(array(0L, dim = dim(volume)))

  thresh <- quantile(pos_voxels, threshold_quantile, na.rm = TRUE)
  mask <- array(0L, dim = dim(volume))
  mask[volume > thresh] <- 1L

  kernel <- shapeKernel(c(3, 3, 3), type = "diamond")
  mask <- erode(mask, kernel)
  mask <- dilate(mask, kernel)

  mask
}

resample_volume <- function(volume, current_spacing, target_spacing = c(1, 1, 1)) {
  if (all(abs(current_spacing - target_spacing) < 0.01)) {
    return(volume)
  }

  current_dims <- dim(volume)
  scale_factors <- current_spacing / target_spacing
  new_dims <- round(current_dims * scale_factors)

  resampled <- array(0, dim = new_dims)

  for (z in seq_len(new_dims[3])) {
    src_z <- min(round(z / scale_factors[3]), current_dims[3])
    src_z <- max(1, src_z)
    for (y in seq_len(new_dims[2])) {
      src_y <- min(round(y / scale_factors[2]), current_dims[2])
      src_y <- max(1, src_y)
      for (x in seq_len(new_dims[1])) {
        src_x <- min(round(x / scale_factors[1]), current_dims[1])
        src_x <- max(1, src_x)
        resampled[x, y, z] <- volume[src_x, src_y, src_z]
      }
    }
  }

  resampled
}

crop_to_brain <- function(volume, mask, margin = 5L) {
  coords <- which(mask > 0, arr.ind = TRUE)
  if (nrow(coords) == 0) return(list(volume = volume, mask = mask, origin = c(1,1,1)))

  mins <- pmax(apply(coords, 2, min) - margin, 1)
  maxs <- pmin(apply(coords, 2, max) + margin, dim(volume))

  cropped_vol <- volume[mins[1]:maxs[1], mins[2]:maxs[2], mins[3]:maxs[3]]
  cropped_mask <- mask[mins[1]:maxs[1], mins[2]:maxs[2], mins[3]:maxs[3]]

  list(volume = cropped_vol, mask = cropped_mask, origin = mins)
}

pad_or_crop_to_size <- function(volume, target_size, center = NULL) {
  current <- dim(volume)
  result <- array(0, dim = target_size)

  if (is.null(center)) {
    center <- current %/% 2
  }

  half <- target_size %/% 2
  src_start <- pmax(center - half + 1L, 1L)
  src_end <- pmin(src_start + target_size - 1L, current)
  src_start <- pmax(src_end - target_size + 1L, 1L)
  actual_size <- src_end - src_start + 1L

  dst_start <- pmax((target_size - actual_size) %/% 2 + 1L, 1L)
  dst_end <- dst_start + actual_size - 1L

  result[dst_start[1]:dst_end[1],
         dst_start[2]:dst_end[2],
         dst_start[3]:dst_end[3]] <-
    volume[src_start[1]:src_end[1],
           src_start[2]:src_end[2],
           src_start[3]:src_end[3]]

  result
}

find_roi_center <- function(seg_mask) {
  coords <- which(seg_mask > 0, arr.ind = TRUE)
  if (nrow(coords) == 0) return(NULL)
  as.integer(round(colMeans(coords)))
}

rigid_registration <- function(moving, fixed) {
  if (!all(dim(moving) == dim(fixed))) {
    moving <- pad_or_crop_to_size(moving, dim(fixed))
  }
  moving
}

preprocess_patient_timepoint <- function(mri_list, target_spacing = c(1, 1, 1),
                                          target_size = c(128, 128, 128),
                                          roi_center = NULL) {
  processed <- list()

  for (mod in names(mri_list$volumes)) {
    vol <- mri_list$volumes[[mod]]
    if (is.null(vol)) next

    vol_data <- as.array(vol)
    brain_mask <- create_brain_mask(vol_data)
    vol_data <- zscore_normalize(vol_data, brain_mask)
    vol_data <- pad_or_crop_to_size(vol_data, target_size, center = roi_center)

    processed[[mod]] <- vol_data
    log_msg("Preprocessed ", mod, " → [", paste(dim(vol_data), collapse = "x"), "]")
  }

  processed
}

preprocess_segmentation <- function(seg_mask, target_size = c(128, 128, 128),
                                     roi_center = NULL) {
  if (is.null(seg_mask)) return(NULL)
  seg_data <- as.array(seg_mask)
  pad_or_crop_to_size(seg_data, target_size, center = roi_center)
}

stack_modalities <- function(processed_volumes, modality_order = c("t1", "t1c", "t2", "flair")) {
  available <- intersect(modality_order, names(processed_volumes))
  if (length(available) == 0) return(NULL)

  target_dim <- dim(processed_volumes[[available[1]]])

  stack <- array(0, dim = c(target_dim, length(modality_order)))
  for (i in seq_along(modality_order)) {
    mod <- modality_order[i]
    if (mod %in% names(processed_volumes)) {
      stack[, , , i] <- processed_volumes[[mod]]
    }
  }

  stack
}

preprocess_full_patient <- function(patient_data, target_size = c(128, 128, 128),
                                     output_dir = NULL) {
  pid <- patient_data$patient_id
  log_msg("Preprocessing patient: ", pid)

  results <- list()

  baseline_vol <- NULL

  for (tp_name in names(patient_data$timepoints)) {
    tp <- patient_data$timepoints[[tp_name]]
    log_msg("  Timepoint: ", tp_name)

    roi_center <- NULL
    if (!is.null(tp$segmentation$mask)) {
      roi_center <- find_roi_center(as.array(tp$segmentation$mask))
    }

    processed <- preprocess_patient_timepoint(tp$mri, target_size = target_size,
                                                roi_center = roi_center)

    if (is.null(baseline_vol) && length(processed) > 0) {
      baseline_vol <- processed[[1]]
    } else if (!is.null(baseline_vol) && length(processed) > 0) {
      for (mod in names(processed)) {
        processed[[mod]] <- rigid_registration(processed[[mod]], baseline_vol)
      }
    }

    seg <- preprocess_segmentation(tp$segmentation$mask, target_size = target_size,
                                    roi_center = roi_center)
    stacked <- stack_modalities(processed)

    results[[tp_name]] <- list(
      stacked = stacked,
      segmentation = seg,
      modalities_available = names(processed)
    )

    if (!is.null(output_dir)) {
      ensure_dir(file.path(output_dir, pid, tp_name))
      if (!is.null(stacked)) {
        saveRDS(stacked, file.path(output_dir, pid, tp_name, "stacked_mri.rds"))
      }
      if (!is.null(seg)) {
        saveRDS(seg, file.path(output_dir, pid, tp_name, "segmentation.rds"))
      }
    }
  }

  log_msg("Preprocessing complete for ", pid)
  results
}
