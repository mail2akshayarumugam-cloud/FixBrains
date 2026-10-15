library(RNifti)

out_base <- file.path(Sys.getenv("USERPROFILE"),
                      "OneDrive/Desktop/test_brain_mri")

create_synthetic_patient <- function(pid, tumor_center, tumor_sizes,
                                     weeks, nx = 96L, ny = 96L, nz = 48L) {
  out_dir <- file.path(out_base, pid)

  coords <- as.matrix(expand.grid(x = seq_len(nx),
                                   y = seq_len(ny),
                                   z = seq_len(nz)))
  cx <- nx / 2; cy <- ny / 2; cz <- nz / 2
  brain_dist <- sqrt((coords[,1]-cx)^2 +
                     (coords[,2]-cy)^2 +
                     ((coords[,3]-cz)*2)^2)
  brain_mask <- brain_dist < 42

  tx <- tumor_center[1]; ty <- tumor_center[2]; tz <- tumor_center[3]
  tumor_dist <- sqrt((coords[,1]-tx)^2 +
                     (coords[,2]-ty)^2 +
                     (coords[,3]-tz)^2)

  for (i in seq_along(weeks)) {
    wname <- weeks[i]
    ts <- tumor_sizes[i]
    wdir <- file.path(out_dir, wname)
    dir.create(wdir, recursive = TRUE, showWarnings = FALSE)

    set.seed(as.integer(chartr("-", "0", gsub("week-", "", wname))) + i * 100)

    brain <- array(rnorm(nx*ny*nz, 1000, 200), dim = c(nx, ny, nz))
    brain[!brain_mask] <- runif(sum(!brain_mask), 0, 50)

    seg <- array(0L, dim = c(nx, ny, nz))
    seg[tumor_dist < ts * 0.4] <- 1L
    seg[tumor_dist >= ts * 0.4 & tumor_dist < ts * 0.7] <- 3L
    seg[tumor_dist >= ts * 0.7 & tumor_dist < ts] <- 2L

    brain[seg == 1L] <- rnorm(sum(seg == 1L), 400, 100)
    brain[seg == 3L] <- rnorm(sum(seg == 3L), 2500, 300)
    brain[seg == 2L] <- rnorm(sum(seg == 2L), 1800, 200)

    t1 <- brain + rnorm(nx*ny*nz, 0, 50)
    t1c <- brain * 1.1; t1c[seg == 3L] <- t1c[seg == 3L] * 1.5
    t2 <- brain * 0.8; t2[seg == 2L] <- t2[seg == 2L] * 1.3
    flair <- brain * 0.9; flair[seg == 2L] <- flair[seg == 2L] * 1.4

    writeNifti(t1, file.path(wdir, "T1.nii.gz"))
    writeNifti(t1c, file.path(wdir, "CT1.nii.gz"))
    writeNifti(t2, file.path(wdir, "T2.nii.gz"))
    writeNifti(flair, file.path(wdir, "FLAIR.nii.gz"))

    seg_dir <- file.path(wdir, "DeepBraTumIA-segmentation",
                         "atlas", "segmentation")
    dir.create(seg_dir, recursive = TRUE, showWarnings = FALSE)
    writeNifti(seg, file.path(seg_dir, "seg_mask.nii.gz"))

    cat("  ", wname, "- tumor radius:", ts, "\n")
  }
  cat("Created:", pid, "\n\n")
}

cat("Creating synthetic test patients...\n\n")

create_synthetic_patient(
  "Synthetic-PD-001",
  tumor_center = c(60, 50, 24),
  tumor_sizes = c(8, 12, 18),
  weeks = c("week-000", "week-012", "week-024"))

create_synthetic_patient(
  "Synthetic-PR-001",
  tumor_center = c(50, 55, 25),
  tumor_sizes = c(16, 12, 7),
  weeks = c("week-000", "week-010", "week-020"))

create_synthetic_patient(
  "Synthetic-SD-001",
  tumor_center = c(55, 48, 22),
  tumor_sizes = c(10, 11, 10),
  weeks = c("week-000", "week-015", "week-030"))

create_synthetic_patient(
  "Synthetic-CR-001",
  tumor_center = c(45, 52, 26),
  tumor_sizes = c(14, 6, 1),
  weeks = c("week-000", "week-008", "week-020"))

create_synthetic_patient(
  "Synthetic-PD-002",
  tumor_center = c(65, 45, 20),
  tumor_sizes = c(5, 10, 15, 22),
  weeks = c("week-000", "week-006", "week-012", "week-024"))

cat("Done! 5 synthetic patients created at:\n")
cat(out_base, "\n\n")
cat("Expected classifications:\n")
cat("  Synthetic-PD-001: Progressive Disease (growing 8→12→18)\n")
cat("  Synthetic-PR-001: Partial Response (shrinking 16→12→7)\n")
cat("  Synthetic-SD-001: Stable Disease (steady ~10)\n")
cat("  Synthetic-CR-001: Complete Response (disappearing 14→6→1)\n")
cat("  Synthetic-PD-002: Progressive Disease (aggressive 5→10→15→22)\n")
