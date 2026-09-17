setwd("C:/Users/mail2/OneDrive/Desktop/R project 1")
library(torch)
library(tidyverse)

TRAINING_DIR <- "data/training"
manifest <- read_csv(file.path(TRAINING_DIR, "manifest.csv"), show_col_types = FALSE)
cat("Converting", nrow(manifest), "samples to Python-compatible format...\n\n")

write_npy <- function(arr, path, dtype = "float32") {
  shape <- dim(arr)
  if (is.null(shape)) shape <- length(arr)

  if (dtype == "float32") {
    descr <- "'<f4'"
    raw_data <- as.numeric(arr)
    byte_size <- 4L
  } else {
    descr <- "'|u1'"
    raw_data <- as.integer(arr)
    byte_size <- 1L
  }

  shape_str <- if (length(shape) == 1) {
    paste0("(", shape, ",)")
  } else {
    paste0("(", paste(rev(shape), collapse = ", "), ",)")
  }

  header <- paste0("{'descr': ", descr,
                    ", 'fortran_order': True",
                    ", 'shape': ", shape_str, ", }")

  header_len <- nchar(header) + 1
  pad_len <- 64 - ((10 + header_len) %% 64)
  if (pad_len == 64) pad_len <- 0
  header <- paste0(header, paste(rep(" ", pad_len), collapse = ""), "\n")

  con <- file(path, "wb")
  writeBin(charToRaw("\x93NUMPY"), con)
  writeBin(as.raw(c(1L, 0L)), con)
  header_bytes <- nchar(header)
  writeBin(as.raw(c(header_bytes %% 256L, header_bytes %/% 256L)), con)
  writeChar(header, con, eos = NULL)

  if (dtype == "float32") {
    writeBin(raw_data, con, size = 4L)
  } else {
    writeBin(as.raw(raw_data), con)
  }
  close(con)
}

mri_out <- file.path(TRAINING_DIR, "mri_npy")
seg_out <- file.path(TRAINING_DIR, "seg_npy")
dir.create(mri_out, showWarnings = FALSE, recursive = TRUE)
dir.create(seg_out, showWarnings = FALSE, recursive = TRUE)

for (i in seq_len(nrow(manifest))) {
  row <- manifest[i, ]
  base_name <- tools::file_path_sans_ext(basename(row$mri_file))

  mri_npy_path <- file.path(mri_out, paste0(base_name, ".npy"))
  seg_npy_path <- file.path(seg_out, paste0(base_name, ".npy"))

  if (file.exists(mri_npy_path) && file.exists(seg_npy_path)) {
    if (i %% 50 == 0) cat("[", i, "/", nrow(manifest), "] skipping (exists)\n")
    next
  }

  mri_tensor <- torch_load(file.path(TRAINING_DIR, row$mri_file))
  seg_tensor <- torch_load(file.path(TRAINING_DIR, row$seg_file))

  mri_arr <- as.array(mri_tensor$to(dtype = torch_float())$cpu())
  seg_arr <- as.array(seg_tensor$to(dtype = torch_int())$cpu())

  write_npy(mri_arr, mri_npy_path, "float32")
  write_npy(seg_arr, seg_npy_path, "uint8")

  if (i %% 10 == 0 || i == nrow(manifest)) {
    cat("[", i, "/", nrow(manifest), "]", base_name, "\n")
  }

  if (i %% 50 == 0) gc()
}

cat("\n========================================\n")
cat("Conversion complete!\n")
cat("MRI files:", mri_out, "\n")
cat("SEG files:", seg_out, "\n")
cat("Upload data/training/ folder to Google Drive\n")
cat("========================================\n")
