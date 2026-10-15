setwd("C:/Users/mail2/OneDrive/Desktop/R project 1")

suppressPackageStartupMessages({
  library(tidyverse)
  library(RNifti)
  library(torch)
  library(data.table)
  library(mmand)
})

source("R/utils.R")
source("R/01_data_loader.R")
source("R/02_preprocessing.R")
source("R/03_model_3d_unet.R")
source("R/04_region_extractor.R")
source("R/11_inference.R")

model <- jit_load("models/unet_3d_trained_jit.pt")

IMAGING <- "data/raw/LUMIERE/Imaging"
test_patients <- c(
  "Patient-083", "Patient-085", "Patient-007",
  "Patient-025", "Patient-042", "Patient-055",
  "Patient-077", "Patient-028", "Patient-041",
  "Patient-091"
)

results <- data.frame(
  patient = character(),
  dice_overall = numeric(),
  dice_necrosis = numeric(),
  dice_edema = numeric(),
  dice_enhancing = numeric(),
  stringsAsFactors = FALSE
)

cat("\n")
cat(strrep("=", 62), "\n")
cat("  3D U-Net v2 — SEGMENTATION ACCURACY\n")
cat(strrep("=", 62), "\n\n")
cat(sprintf(
  "%-14s | %-8s | %-8s | %-8s | %-8s\n",
  "Patient", "Overall", "Necrosis", "Edema", "Enhanc."
))
cat(strrep("-", 62), "\n")

for (pid in test_patients) {
  pp <- file.path(IMAGING, pid)
  if (!dir.exists(pp)) next

  pd <- tryCatch(load_full_patient(pp), error = function(e) NULL)
  if (is.null(pd)) next

  pre <- tryCatch(
    preprocess_full_patient(pd, target_size = c(48L, 48L, 48L)),
    error = function(e) NULL
  )
  if (is.null(pre)) next

  tp1 <- names(pre)[1]
  tp <- pre[[tp1]]
  if (is.null(tp$stacked) || is.null(tp$segmentation)) next

  gt <- tp$segmentation
  pred <- segment_mri(model, tp$stacked, "cpu",
                      postprocess = TRUE, use_tta = FALSE)

  dice_nec <- compute_dice(pred == 1, gt == 1)
  dice_ede <- compute_dice(pred == 2, gt == 2)
  dice_enh <- compute_dice(pred == 3, gt == 3)
  dice_avg <- mean(c(dice_nec, dice_ede, dice_enh), na.rm = TRUE)

  results <- rbind(results, data.frame(
    patient = pid,
    dice_overall = dice_avg,
    dice_necrosis = dice_nec,
    dice_edema = dice_ede,
    dice_enhancing = dice_enh
  ))

  cat(sprintf(
    "%-14s | %-8.4f | %-8.4f | %-8.4f | %-8.4f\n",
    pid, dice_avg, dice_nec, dice_ede, dice_enh
  ))

  rm(pd, pre, gt, pred)
  gc()
}

cat(strrep("-", 62), "\n")
cat(sprintf(
  "%-14s | %-8.4f | %-8.4f | %-8.4f | %-8.4f\n",
  "AVERAGE",
  mean(results$dice_overall, na.rm = TRUE),
  mean(results$dice_necrosis, na.rm = TRUE),
  mean(results$dice_edema, na.rm = TRUE),
  mean(results$dice_enhancing, na.rm = TRUE)
))

cat("\n")
cat(strrep("=", 62), "\n")
cat("  PROGRESSION CLASSIFICATION ACCURACY\n")
cat(strrep("=", 62), "\n\n")

synth_dir <- file.path(
  Sys.getenv("USERPROFILE"),
  "OneDrive/Desktop/test_brain_mri"
)

synth_patients <- c(
  "Synthetic-PD-001", "Synthetic-PR-001",
  "Synthetic-SD-001", "Synthetic-CR-001",
  "Synthetic-PD-002"
)
expected <- c("PD", "PR", "SD", "CR", "PD")

source("R/05_region_tracker.R")
source("R/06_feature_extraction.R")
source("R/07_progression_engine.R")

correct <- 0
total <- 0

cat(sprintf(
  "%-20s | %-8s | %-8s | %-6s | %s\n",
  "Patient", "Expected", "Got", "Conf", "Result"
))
cat(strrep("-", 62), "\n")

for (i in seq_along(synth_patients)) {
  sp <- synth_patients[i]
  pp <- file.path(synth_dir, sp)
  if (!dir.exists(pp)) next

  pd <- tryCatch(load_full_patient(pp), error = function(e) NULL)
  if (is.null(pd)) next

  pre <- tryCatch(
    preprocess_full_patient(pd, target_size = c(48L, 48L, 48L)),
    error = function(e) NULL
  )
  if (is.null(pre)) next

  seg <- segment_patient(model, pre, "cpu")
  rd <- extract_all_patient_regions(seg, sp, min_vol = 50)

  if (length(rd$regions_by_timepoint) < 2) next

  tr <- track_patient_regions(rd$regions_by_timepoint)
  ev <- create_event_table(tr$all_events, sp)
  tps <- names(rd$regions_by_timepoint)
  tw <- sapply(pd$timepoints[tps], function(x) x$week_number)

  trajs <- list()
  for (r in rd$regions_by_timepoint[[1]]) {
    vols <- sapply(tps, function(tp) {
      m <- Filter(
        function(x) x$region_type == r$region_type,
        rd$regions_by_timepoint[[tp]]
      )
      if (length(m) > 0) m[[1]]$volume_cm3 else 0
    })
    trajs[[r$region_id]] <- analyze_region_trajectory(vols, tw)
  }

  prog <- classify_patient_progression(trajs, ev)
  match <- prog$classification == expected[i]
  correct <- correct + match
  total <- total + 1

  cat(sprintf(
    "%-20s | %-8s | %-8s | %-5d%% | %s\n",
    sp, expected[i], prog$classification,
    round(prog$confidence * 100),
    if (match) "CORRECT" else "WRONG"
  ))

  rm(pd, pre, seg, rd)
  gc()
}

cat(strrep("-", 62), "\n")
cat(sprintf("  Accuracy: %d/%d = %.0f%%\n", correct, total,
            correct / total * 100))

cat("\n")
cat(strrep("=", 62), "\n")
cat("  SUMMARY\n")
cat(strrep("=", 62), "\n")
cat("  Segmentation (Dice):\n")
cat("    Overall:   ", round(mean(results$dice_overall, na.rm = TRUE), 4), "\n")
cat("    Necrosis:  ", round(mean(results$dice_necrosis, na.rm = TRUE), 4), "\n")
cat("    Edema:     ", round(mean(results$dice_edema, na.rm = TRUE), 4), "\n")
cat("    Enhancing: ", round(mean(results$dice_enhancing, na.rm = TRUE), 4), "\n")
cat("  Classification:\n")
cat("    Synthetic test:", correct, "/", total, "correct\n")
cat("    LUMIERE ablation: 72.7% (region-wise)\n")
cat(strrep("=", 62), "\n")

write_csv(results, "results/tables/model_evaluation.csv")
cat("\nSaved to results/tables/model_evaluation.csv\n")
