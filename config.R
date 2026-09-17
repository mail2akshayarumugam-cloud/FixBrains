# ──────────────────────────────────────────────────────────────
# Project Configuration
# ──────────────────────────────────────────────────────────────

PROJECT_ROOT <- normalizePath(dirname(sys.frame(1)$ofile), winslash = "/")

PATHS <- list(
  raw_data     = file.path(PROJECT_ROOT, "data", "raw", "LUMIERE"),
  processed    = file.path(PROJECT_ROOT, "data", "processed"),
  registered   = file.path(PROJECT_ROOT, "data", "processed", "registered"),
  features     = file.path(PROJECT_ROOT, "data", "processed", "features"),
  regions      = file.path(PROJECT_ROOT, "data", "processed", "region_tables"),
  slices       = file.path(PROJECT_ROOT, "data", "processed", "slices"),
  models       = file.path(PROJECT_ROOT, "models"),
  results      = file.path(PROJECT_ROOT, "results"),
  figures      = file.path(PROJECT_ROOT, "results", "figures"),
  tables       = file.path(PROJECT_ROOT, "results", "tables"),
  reports      = file.path(PROJECT_ROOT, "results", "patient_reports")
)

SEGMENTATION_LABELS <- list(
  background = 0L,
  necrosis   = 1L,
  edema      = 2L,
  enhancing  = 3L
)

RANO_LEVELS <- c("CR", "PR", "SD", "PD")

RISK_THRESHOLDS <- list(
  growth_significant   = 0.25,
  shrinkage_significant = -0.25,
  iou_match_threshold  = 0.15,
  centroid_max_mm      = 30,
  min_region_volume_mm3 = 100
)

UNET_CONFIG <- list(
  input_shape    = c(128L, 128L, 128L, 4L),
  n_classes      = 4L,
  base_filters   = 16L,
  learning_rate  = 1e-4,
  batch_size     = 2L,
  epochs         = 50L,
  patch_size     = c(128L, 128L, 128L)
)

LSTM_CONFIG <- list(
  hidden_size    = 64L,
  num_layers     = 2L,
  dropout        = 0.3,
  learning_rate  = 1e-3,
  batch_size     = 16L,
  epochs         = 100L,
  n_classes      = 4L
)
