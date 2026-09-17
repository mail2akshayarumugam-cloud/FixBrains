# ──────────────────────────────────────────────────────────────
# Shared Utilities
# ──────────────────────────────────────────────────────────────

suppressPackageStartupMessages(library(RNifti))

UTILS_LOADED <- TRUE

ensure_dir <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE)
  invisible(path)
}

get_voxel_volume_mm3 <- function(img) {
  spacing <- RNifti::pixdim(img)[2:4]
  prod(abs(spacing))
}

get_voxel_spacing <- function(img) {
  abs(RNifti::pixdim(img)[2:4])
}

mm3_to_cm3 <- function(vol_mm3) {
  vol_mm3 / 1000
}

euclidean_distance <- function(p1, p2) {
  sqrt(sum((p1 - p2)^2))
}

compute_centroid <- function(mask, spacing = c(1, 1, 1)) {
  coords <- which(mask > 0, arr.ind = TRUE)
  if (nrow(coords) == 0) return(c(NA, NA, NA))
  scaled <- sweep(coords, 2, spacing, `*`)
  colMeans(scaled)
}

compute_bounding_box <- function(mask) {
  coords <- which(mask > 0, arr.ind = TRUE)
  if (nrow(coords) == 0) return(NULL)
  list(
    min = apply(coords, 2, min),
    max = apply(coords, 2, max),
    size = apply(coords, 2, max) - apply(coords, 2, min) + 1
  )
}

compute_iou <- function(mask_a, mask_b) {
  a <- mask_a > 0
  b <- mask_b > 0
  intersection <- sum(a & b)
  union <- sum(a | b)
  if (union == 0) return(0)
  intersection / union
}

compute_dice <- function(mask_a, mask_b) {
  a <- mask_a > 0
  b <- mask_b > 0
  intersection <- sum(a & b)
  total <- sum(a) + sum(b)
  if (total == 0) return(0)
  2 * intersection / total
}

normalize_intensity <- function(vol, mask = NULL) {
  if (!is.null(mask)) {
    vals <- vol[mask > 0]
  } else {
    vals <- vol[vol > 0]
  }
  if (length(vals) == 0) return(vol)
  mu <- mean(vals)
  sigma <- sd(vals)
  if (sigma < 1e-8) return(vol * 0)
  (vol - mu) / sigma
}

extract_middle_slice <- function(vol, axis = 3) {
  dims <- dim(vol)
  mid <- dims[axis] %/% 2
  switch(axis,
    vol[mid, , ],
    vol[, mid, ],
    vol[, , mid]
  )
}

safe_read_nifti <- function(path) {
  if (!file.exists(path)) {
    warning("NIfTI file not found: ", path)
    return(NULL)
  }
  tryCatch(
    RNifti::readNifti(path),
    error = function(e) {
      warning("Failed to read NIfTI: ", path, " — ", e$message)
      NULL
    }
  )
}

log_msg <- function(..., level = "INFO") {
  ts <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  cat(sprintf("[%s] %s: %s\n", ts, level, paste0(...)))
}
