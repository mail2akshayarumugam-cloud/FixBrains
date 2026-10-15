# ──────────────────────────────────────────────────────────────
# Stage 6: Quantitative Feature Extraction
# Per region, per timepoint: volume, growth, shape, spatial
# ──────────────────────────────────────────────────────────────

library(tidyverse)

if (!exists("UTILS_LOADED", envir = .GlobalEnv)) {
  SCRIPT_DIR <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) "R")
  source(file.path(SCRIPT_DIR, "utils.R"))
}

compute_surface_area <- function(mask, spacing = c(1, 1, 1)) {
  binary <- (mask > 0)
  if (sum(binary) == 0) return(0)

  dims <- dim(binary)
  padded <- array(FALSE, dim = dims + 2L)
  padded[2:(dims[1]+1), 2:(dims[2]+1), 2:(dims[3]+1)] <- binary

  face_xp <- sum(binary & !padded[3:(dims[1]+2), 2:(dims[2]+1), 2:(dims[3]+1)])
  face_xn <- sum(binary & !padded[1:dims[1],     2:(dims[2]+1), 2:(dims[3]+1)])
  face_yp <- sum(binary & !padded[2:(dims[1]+1), 3:(dims[2]+2), 2:(dims[3]+1)])
  face_yn <- sum(binary & !padded[2:(dims[1]+1), 1:dims[2],     2:(dims[3]+1)])
  face_zp <- sum(binary & !padded[2:(dims[1]+1), 2:(dims[2]+1), 3:(dims[3]+2)])
  face_zn <- sum(binary & !padded[2:(dims[1]+1), 2:(dims[2]+1), 1:dims[3]])

  area_yz <- spacing[2] * spacing[3]
  area_xz <- spacing[1] * spacing[3]
  area_xy <- spacing[1] * spacing[2]

  (face_xp + face_xn) * area_yz + (face_yp + face_yn) * area_xz + (face_zp + face_zn) * area_xy
}

compute_shape_features <- function(mask, spacing = c(1, 1, 1)) {
  n_voxels <- sum(mask > 0)
  if (n_voxels == 0) {
    return(list(
      volume_mm3 = 0, surface_area = 0, compactness = NA,
      sphericity = NA, elongation = NA
    ))
  }

  voxel_vol <- prod(spacing)
  volume <- n_voxels * voxel_vol

  sa <- compute_surface_area(mask, spacing)

  sphericity <- if (sa > 0) {
    (pi^(1/3) * (6 * volume)^(2/3)) / sa
  } else NA

  compactness <- if (sa > 0) {
    (36 * pi * volume^2) / (sa^3)
  } else NA

  bbox <- compute_bounding_box(mask)
  bbox_dims <- sort(bbox$size * spacing, decreasing = TRUE)
  elongation <- if (bbox_dims[1] > 0) bbox_dims[2] / bbox_dims[1] else NA

  list(
    volume_mm3 = volume,
    surface_area = sa,
    compactness = compactness,
    sphericity = sphericity,
    elongation = elongation,
    bbox_dims = bbox_dims
  )
}

compute_intensity_features <- function(mri_volume, mask) {
  if (is.null(mri_volume) || sum(mask > 0) == 0) {
    return(list(
      mean_intensity = NA, sd_intensity = NA,
      median_intensity = NA, skewness = NA,
      kurtosis = NA, intensity_range = NA
    ))
  }

  vals <- mri_volume[mask > 0]

  n <- length(vals)
  mu <- mean(vals)
  sigma <- sd(vals)
  skew <- if (sigma > 0) mean(((vals - mu) / sigma)^3) else 0
  kurt <- if (sigma > 0) mean(((vals - mu) / sigma)^4) - 3 else 0

  list(
    mean_intensity = mu,
    sd_intensity = sigma,
    median_intensity = median(vals),
    skewness = skew,
    kurtosis = kurt,
    intensity_range = diff(range(vals))
  )
}

extract_region_features <- function(region, spacing = c(1, 1, 1), mri_volume = NULL) {
  shape <- compute_shape_features(region$mask, spacing)
  centroid <- compute_centroid(region$mask, spacing)
  intensity <- compute_intensity_features(mri_volume, region$mask)

  c(
    list(
      region_id = region$region_id,
      region_type = region$region_type,
      centroid_x = centroid[1],
      centroid_y = centroid[2],
      centroid_z = centroid[3]
    ),
    shape,
    intensity
  )
}

compute_longitudinal_features <- function(region_table, tracking_events) {
  patients <- unique(region_table$patient_id)
  all_features <- list()

  for (pid in patients) {
    pt <- region_table %>% filter(patient_id == pid)
    timepoints <- sort(unique(pt$timepoint))

    if (length(timepoints) < 2) next

    continuations <- tracking_events %>%
      filter(patient_id == pid, event_type == "CONTINUATION")

    for (i in seq_len(nrow(continuations))) {
      cont <- continuations[i, ]
      r1 <- pt %>% filter(timepoint == cont$from_tp, region_id == cont$from_region)
      r2 <- pt %>% filter(timepoint == cont$to_tp, region_id == cont$to_region)

      if (nrow(r1) == 0 || nrow(r2) == 0) next

      abs_change <- r2$volume_cm3 - r1$volume_cm3
      pct_change <- if (r1$volume_cm3 > 1e-8) abs_change / r1$volume_cm3 else NA_real_

      centroid_disp <- euclidean_distance(
        c(r1$centroid_x, r1$centroid_y, r1$centroid_z),
        c(r2$centroid_x, r2$centroid_y, r2$centroid_z)
      )

      all_features <- c(all_features, list(tibble(
        patient_id = pid,
        from_tp = cont$from_tp,
        to_tp = cont$to_tp,
        region_id = cont$from_region,
        region_type = r1$region_type,
        vol_t1 = r1$volume_cm3,
        vol_t2 = r2$volume_cm3,
        abs_volume_change = abs_change,
        pct_volume_change = pct_change,
        centroid_displacement = centroid_disp
      )))
    }
  }

  if (length(all_features) == 0) {
    return(tibble())
  }

  bind_rows(all_features)
}

compute_patient_summary_features <- function(region_table, event_table, long_features) {
  patients <- unique(region_table$patient_id)

  map_dfr(patients, function(pid) {
    pt_regions <- region_table %>% filter(patient_id == pid)
    pt_events <- event_table %>% filter(patient_id == pid)
    pt_long <- long_features %>% filter(patient_id == pid)

    n_timepoints <- length(unique(pt_regions$timepoint))
    n_regions_total <- nrow(pt_regions)

    total_vol_first <- pt_regions %>%
      filter(timepoint == min(timepoint)) %>%
      summarise(v = sum(volume_cm3, na.rm = TRUE)) %>%
      pull(v)

    total_vol_last <- pt_regions %>%
      filter(timepoint == max(timepoint)) %>%
      summarise(v = sum(volume_cm3, na.rm = TRUE)) %>%
      pull(v)

    total_vol_change <- total_vol_last - total_vol_first
    total_vol_pct <- if (total_vol_first > 0) total_vol_change / total_vol_first else NA

    n_growing <- if (nrow(pt_long) > 0) sum(pt_long$pct_volume_change > 0.25, na.rm = TRUE) else 0
    n_shrinking <- if (nrow(pt_long) > 0) sum(pt_long$pct_volume_change < -0.25, na.rm = TRUE) else 0

    max_growth_rate <- if (nrow(pt_long) > 0 && any(!is.na(pt_long$pct_volume_change))) {
      max(pt_long$pct_volume_change, na.rm = TRUE)
    } else 0
    max_centroid_disp <- if (nrow(pt_long) > 0 && any(!is.na(pt_long$centroid_displacement))) {
      max(pt_long$centroid_displacement, na.rm = TRUE)
    } else 0

    n_appearances <- sum(pt_events$event_type == "APPEARANCE")
    n_disappearances <- sum(pt_events$event_type == "DISAPPEARANCE")
    n_splits <- sum(pt_events$event_type == "SPLIT")
    n_merges <- sum(pt_events$event_type == "MERGE")

    tibble(
      patient_id = pid,
      n_timepoints = n_timepoints,
      n_regions_total = n_regions_total,
      total_vol_first = total_vol_first,
      total_vol_last = total_vol_last,
      total_vol_change = total_vol_change,
      total_vol_pct_change = total_vol_pct,
      n_growing_regions = n_growing,
      n_shrinking_regions = n_shrinking,
      max_growth_rate = max_growth_rate,
      max_centroid_displacement = max_centroid_disp,
      n_appearances = n_appearances,
      n_disappearances = n_disappearances,
      n_splits = n_splits,
      n_merges = n_merges,
      total_events = n_appearances + n_disappearances + n_splits + n_merges
    )
  })
}
