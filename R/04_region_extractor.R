# ──────────────────────────────────────────────────────────────
# Stage 4: Region Extraction & Standardization
# Connected component analysis on segmentation masks
# ──────────────────────────────────────────────────────────────

library(mmand)
library(tidyverse)

if (!exists("UTILS_LOADED", envir = .GlobalEnv)) {
  SCRIPT_DIR <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) "R")
  source(file.path(SCRIPT_DIR, "utils.R"))
}

label_connected_components_3d <- function(binary_mask) {
  dims <- dim(binary_mask)
  labels <- array(0L, dim = dims)

  nz <- sum(binary_mask > 0)
  if (nz == 0) return(list(labels = labels, n_components = 0L))

  current_label <- 0L
  queue_x <- integer(nz)
  queue_y <- integer(nz)
  queue_z <- integer(nz)

  dx <- c(-1L, 1L, 0L, 0L, 0L, 0L)
  dy <- c(0L, 0L, -1L, 1L, 0L, 0L)
  dz <- c(0L, 0L, 0L, 0L, -1L, 1L)

  for (z in seq_len(dims[3])) {
    for (y in seq_len(dims[2])) {
      for (x in seq_len(dims[1])) {
        if (binary_mask[x, y, z] > 0 && labels[x, y, z] == 0L) {
          current_label <- current_label + 1L
          labels[x, y, z] <- current_label
          head_ptr <- 1L
          tail_ptr <- 1L
          queue_x[1] <- x; queue_y[1] <- y; queue_z[1] <- z

          while (head_ptr <= tail_ptr) {
            cx <- queue_x[head_ptr]
            cy <- queue_y[head_ptr]
            cz <- queue_z[head_ptr]
            head_ptr <- head_ptr + 1L

            for (d in seq_len(6L)) {
              nx <- cx + dx[d]
              ny <- cy + dy[d]
              nz2 <- cz + dz[d]
              if (nx >= 1L && nx <= dims[1] &&
                  ny >= 1L && ny <= dims[2] &&
                  nz2 >= 1L && nz2 <= dims[3] &&
                  binary_mask[nx, ny, nz2] > 0 &&
                  labels[nx, ny, nz2] == 0L) {
                labels[nx, ny, nz2] <- current_label
                tail_ptr <- tail_ptr + 1L
                queue_x[tail_ptr] <- nx
                queue_y[tail_ptr] <- ny
                queue_z[tail_ptr] <- nz2
              }
            }
          }
        }
      }
    }
  }

  list(labels = labels, n_components = current_label)
}

extract_regions_from_segmentation <- function(seg_mask, spacing = c(1, 1, 1),
                                               min_volume_mm3 = 100) {
  seg_data <- as.array(seg_mask)
  unique_labels <- sort(unique(as.integer(seg_data)))
  unique_labels <- unique_labels[unique_labels > 0]

  voxel_vol <- prod(spacing)
  regions <- list()
  region_id <- 0L

  label_names <- c("1" = "necrosis", "2" = "edema", "3" = "enhancing",
                    "4" = "enhancing")

  for (lbl in unique_labels) {
    binary <- array(0L, dim = dim(seg_data))
    binary[seg_data == lbl] <- 1L

    cc <- label_connected_components_3d(binary)

    for (comp in seq_len(cc$n_components)) {
      comp_mask <- array(0L, dim = dim(seg_data))
      comp_mask[cc$labels == comp] <- 1L

      volume_mm3 <- sum(comp_mask) * voxel_vol
      if (volume_mm3 < min_volume_mm3) next

      region_id <- region_id + 1L
      centroid <- compute_centroid(comp_mask, spacing)
      bbox <- compute_bounding_box(comp_mask)

      region_type <- if (as.character(lbl) %in% names(label_names)) {
        label_names[[as.character(lbl)]]
      } else {
        paste0("label_", lbl)
      }

      regions[[region_id]] <- list(
        region_id = sprintf("R%03d", region_id),
        seg_label = lbl,
        region_type = region_type,
        mask = comp_mask,
        volume_mm3 = volume_mm3,
        volume_cm3 = mm3_to_cm3(volume_mm3),
        centroid = centroid,
        bounding_box = bbox,
        n_voxels = sum(comp_mask)
      )
    }
  }

  log_msg("Extracted ", length(regions), " regions (min vol: ", min_volume_mm3, " mm³)")
  regions
}

create_region_table <- function(regions, patient_id, timepoint) {
  if (length(regions) == 0) {
    return(tibble(
      patient_id = character(0), timepoint = character(0),
      region_id = character(0), region_type = character(0),
      volume_mm3 = numeric(0), volume_cm3 = numeric(0),
      centroid_x = numeric(0), centroid_y = numeric(0), centroid_z = numeric(0),
      bbox_size_x = integer(0), bbox_size_y = integer(0), bbox_size_z = integer(0),
      n_voxels = integer(0)
    ))
  }

  map_dfr(regions, function(r) {
    bb <- r$bounding_box
    tibble(
      patient_id = patient_id,
      timepoint = timepoint,
      region_id = r$region_id,
      seg_label = r$seg_label,
      region_type = r$region_type,
      volume_mm3 = r$volume_mm3,
      volume_cm3 = r$volume_cm3,
      centroid_x = r$centroid[1],
      centroid_y = r$centroid[2],
      centroid_z = r$centroid[3],
      bbox_size_x = if (!is.null(bb)) bb$size[1] else NA_integer_,
      bbox_size_y = if (!is.null(bb)) bb$size[2] else NA_integer_,
      bbox_size_z = if (!is.null(bb)) bb$size[3] else NA_integer_,
      n_voxels = r$n_voxels
    )
  })
}

extract_all_patient_regions <- function(patient_preprocessed, patient_id,
                                         spacing = c(1, 1, 1), min_vol = 100) {
  all_regions <- list()
  all_tables <- list()

  for (tp_name in names(patient_preprocessed)) {
    tp <- patient_preprocessed[[tp_name]]
    if (is.null(tp$segmentation)) {
      log_msg("  Skipping ", tp_name, " — no segmentation", level = "WARN")
      next
    }

    regions <- extract_regions_from_segmentation(
      tp$segmentation, spacing = spacing, min_volume_mm3 = min_vol
    )

    all_regions[[tp_name]] <- regions
    all_tables[[tp_name]] <- create_region_table(regions, patient_id, tp_name)
    log_msg("  ", tp_name, ": ", length(regions), " regions extracted")
  }

  combined_table <- bind_rows(all_tables)

  list(
    regions_by_timepoint = all_regions,
    table = combined_table
  )
}
