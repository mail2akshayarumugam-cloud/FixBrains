# ──────────────────────────────────────────────────────────────
# Stage 1: LUMIERE Data Loader
# Actual LUMIERE structure:
#   Imaging/Patient-XXX/week-YYY/
#     T1.nii.gz, CT1.nii.gz, T2.nii.gz, FLAIR.nii.gz
#     HD-GLIO-AUTO-segmentation/registered/segmentation.nii.gz
#     DeepBraTumIA-segmentation/atlas/segmentation/seg_mask.nii.gz
# ──────────────────────────────────────────────────────────────

library(tidyverse)
library(RNifti)
library(data.table)

if (!exists("UTILS_LOADED", envir = .GlobalEnv)) {
  SCRIPT_DIR <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) "R")
  source(file.path(SCRIPT_DIR, "utils.R"))
}

parse_week_number <- function(dirname) {
  name <- sub("^week-", "", dirname)
  parts <- strsplit(name, "-")[[1]]
  week <- as.numeric(parts[1])
  if (length(parts) > 1) {
    week <- week + as.numeric(parts[2]) * 0.1
  }
  week
}

find_imaging_root <- function(data_path) {
  imaging_dir <- file.path(data_path, "Imaging")
  if (dir.exists(imaging_dir)) return(imaging_dir)
  if (any(grepl("^Patient-", list.dirs(data_path, recursive = FALSE, full.names = FALSE)))) {
    return(data_path)
  }
  subs <- list.dirs(data_path, recursive = FALSE)
  for (s in subs) {
    if (any(grepl("^Patient-", list.dirs(s, recursive = FALSE, full.names = FALSE)))) {
      return(s)
    }
  }
  data_path
}

load_patient_index <- function(data_path) {
  root <- find_imaging_root(data_path)
  log_msg("Imaging root: ", root)

  patient_dirs <- list.dirs(root, recursive = FALSE, full.names = TRUE)
  patient_dirs <- patient_dirs[grepl("Patient-", basename(patient_dirs))]

  patients <- map_dfr(patient_dirs, function(pdir) {
    pid <- basename(pdir)
    week_dirs <- list.dirs(pdir, recursive = FALSE, full.names = TRUE)
    week_dirs <- week_dirs[grepl("^week-", basename(week_dirs))]

    week_nums <- sapply(basename(week_dirs), parse_week_number)

    tibble(
      patient_id = pid,
      n_timepoints = length(week_dirs),
      weeks = list(sort(week_nums)),
      path = pdir
    )
  })

  patients <- patients %>% arrange(patient_id)
  log_msg("Indexed ", nrow(patients), " patients, ",
          sum(patients$n_timepoints), " total timepoints")
  patients
}

load_mri_timepoint <- function(week_path, use_atlas = TRUE) {
  atlas_dir <- file.path(week_path, "DeepBraTumIA-segmentation",
                          "atlas", "skull_strip")

  if (use_atlas && dir.exists(atlas_dir)) {
    modalities <- list(
      t1    = file.path(atlas_dir, "t1_skull_strip.nii.gz"),
      t1c   = file.path(atlas_dir, "ct1_skull_strip.nii.gz"),
      t2    = file.path(atlas_dir, "t2_skull_strip.nii.gz"),
      flair = file.path(atlas_dir, "flair_skull_strip.nii.gz")
    )
    log_msg("    Using atlas skull-stripped volumes")
  } else {
    modalities <- list(
      t1    = file.path(week_path, "T1.nii.gz"),
      t1c   = file.path(week_path, "CT1.nii.gz"),
      t2    = file.path(week_path, "T2.nii.gz"),
      flair = file.path(week_path, "FLAIR.nii.gz")
    )
  }

  volumes <- list()
  available <- c()

  for (mod in names(modalities)) {
    path <- modalities[[mod]]
    if (file.exists(path)) {
      volumes[[mod]] <- safe_read_nifti(path)
      if (!is.null(volumes[[mod]])) {
        available <- c(available, mod)
        dims <- paste(dim(volumes[[mod]]), collapse = "x")
        log_msg("    ", toupper(mod), ": ", dims)
      }
    }
  }

  list(volumes = volumes, files = modalities, available = available)
}

load_segmentation_timepoint <- function(week_path, method = "deepbratumia") {
  if (method == "hdglio") {
    seg_path <- file.path(week_path,
                          "HD-GLIO-AUTO-segmentation", "registered", "segmentation.nii.gz")
  } else {
    seg_path <- file.path(week_path,
                          "DeepBraTumIA-segmentation", "atlas", "segmentation", "seg_mask.nii.gz")
  }

  if (!file.exists(seg_path) && method == "hdglio") {
    seg_path <- file.path(week_path,
                          "DeepBraTumIA-segmentation", "atlas", "segmentation", "seg_mask.nii.gz")
  }

  if (!file.exists(seg_path)) {
    seg_files <- list.files(week_path, pattern = "seg.*\\.nii\\.gz$",
                            recursive = TRUE, full.names = TRUE)
    if (length(seg_files) > 0) seg_path <- seg_files[1]
  }

  if (file.exists(seg_path)) {
    seg <- safe_read_nifti(seg_path)
    if (!is.null(seg)) {
      labels <- sort(unique(as.integer(seg)))
      log_msg("    Seg: ", basename(seg_path), " labels=[",
              paste(labels, collapse = ","), "]")
      return(list(mask = seg, file = seg_path, labels = labels, method = method))
    }
  }

  log_msg("    No segmentation found", level = "WARN")
  list(mask = NULL, file = NA, labels = integer(0), method = NA)
}

load_volume_json <- function(week_path) {
  json_path <- file.path(week_path,
                          "DeepBraTumIA-segmentation", "atlas", "segmentation",
                          "measured_volumes_in_mm3.json")
  if (file.exists(json_path)) {
    return(jsonlite::fromJSON(json_path))
  }
  NULL
}

load_full_patient <- function(patient_path, seg_method = "deepbratumia") {
  pid <- basename(patient_path)
  week_dirs <- list.dirs(patient_path, recursive = FALSE, full.names = TRUE)
  week_dirs <- week_dirs[grepl("^week-", basename(week_dirs))]

  week_nums <- sapply(basename(week_dirs), parse_week_number)
  ord <- order(week_nums)
  week_dirs <- week_dirs[ord]
  week_names <- basename(week_dirs)

  log_msg("Loading ", pid, " — ", length(week_dirs), " timepoints")

  tp_data <- list()
  for (i in seq_along(week_dirs)) {
    wdir <- week_dirs[i]
    wname <- week_names[i]
    log_msg("  ", wname, ":")

    mri <- load_mri_timepoint(wdir)
    seg <- load_segmentation_timepoint(wdir, seg_method)
    vol_json <- load_volume_json(wdir)

    tp_data[[wname]] <- list(
      name = wname,
      week_number = week_nums[ord[i]],
      mri = mri,
      segmentation = seg,
      precomputed_volumes = vol_json,
      path = wdir
    )
  }

  list(patient_id = pid, timepoints = tp_data, path = patient_path)
}

load_clinical_data <- function(data_path) {
  csv_files <- list.files(data_path, pattern = "\\.csv$",
                          recursive = TRUE, full.names = TRUE)
  json_files <- list.files(data_path, pattern = "\\.json$",
                           recursive = TRUE, full.names = TRUE)

  clinical <- list()
  for (f in csv_files) {
    name <- tools::file_path_sans_ext(basename(f))
    tryCatch({
      clinical[[name]] <- fread(f)
      log_msg("Loaded CSV: ", basename(f), " [", nrow(clinical[[name]]), " rows]")
    }, error = function(e) NULL)
  }
  clinical
}

summarize_dataset <- function(data_path) {
  patients <- load_patient_index(data_path)
  clinical <- load_clinical_data(data_path)

  total_tp <- sum(patients$n_timepoints)

  cat("\n══════════════════════════════════════════\n")
  cat("  LUMIERE Dataset Summary\n")
  cat("══════════════════════════════════════════\n")
  cat("  Patients:              ", nrow(patients), "\n")
  cat("  Total timepoints:      ", total_tp, "\n")
  cat("  Avg timepoints/patient:", round(mean(patients$n_timepoints), 1), "\n")
  cat("  Min timepoints:        ", min(patients$n_timepoints), "\n")
  cat("  Max timepoints:        ", max(patients$n_timepoints), "\n")
  cat("  CSV files:             ", length(clinical), "\n")
  cat("══════════════════════════════════════════\n\n")

  list(patients = patients, clinical = clinical)
}
