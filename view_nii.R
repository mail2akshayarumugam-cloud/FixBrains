# View contents of a .nii.gz file
# Usage:
#   Rscript view_nii.R "path/to/file.nii.gz"

suppressPackageStartupMessages(library(RNifti))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  cat("Usage: Rscript view_nii.R <path/to/file.nii.gz>\n")
  quit(status = 1)
}

nii_path <- args[1]

if (!file.exists(nii_path)) {
  stop("File not found: ", nii_path)
}

img <- readNifti(nii_path)

cat("\n=== NIfTI file info ===\n")
cat("Path: ", nii_path, "\n")
cat("Class: ", class(img), "\n")
cat("Dimensions: ", paste(dim(img), collapse = " x "), "\n")
cat("Data type: ", storage.mode(img), "\n")
cat("Min value: ", min(img, na.rm = TRUE), "\n")
cat("Max value: ", max(img, na.rm = TRUE), "\n")
cat("Mean value: ", mean(as.numeric(img), na.rm = TRUE), "\n")
cat("Unique values: ", length(unique(as.numeric(img))), "\n")

spacing <- pixdim(img)
if (length(spacing) >= 4) {
  cat("Voxel spacing: ", paste(abs(spacing[2:4]), collapse = " x "), "\n")
}

# Show a small preview of the middle slice in each orientation
cat("\n=== Slice preview ===\n")

axis3 <- dim(img)[3]
axis2 <- dim(img)[2]
axis1 <- dim(img)[1]

slice_z <- if (axis3 > 1) floor(axis3 / 2) else 1
slice_y <- if (axis2 > 1) floor(axis2 / 2) else 1
slice_x <- if (axis1 > 1) floor(axis1 / 2) else 1

cat("Middle axial slice (z = ", slice_z, "):\n")
print(img[, , slice_z])

cat("\nMiddle sagittal slice (x = ", slice_x, "):\n")
print(img[slice_x, , ])

cat("\nMiddle coronal slice (y = ", slice_y, "):\n")
print(img[, slice_y, ])

cat("\n=== First few voxel values ===\n")
print(as.vector(img)[1:min(20, length(as.vector(img)))])
