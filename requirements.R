# ──────────────────────────────────────────────────────────────
# Install all required packages
# Run once: Rscript requirements.R
# ──────────────────────────────────────────────────────────────

cran_packages <- c(
  # Data
  "tidyverse",
  "data.table",
  "jsonlite",

  # NIfTI / Neuroimaging
  "RNifti",
  "oro.nifti",
  "neurobase",

  # Image processing
  "mmand",
  "imager",

  # 3D Deep Learning

  "torch",
  "luz",
  "torchvision",

  # Statistics & Modeling
  "lme4",
  "caret",
  "nnet",

  # Evaluation
  "irr",
  "pROC",
  "yardstick",

  # Visualization
  "ggplot2",
  "plotly",
  "patchwork",
  "viridis",
  "RColorBrewer",

  # Reporting
  "rmarkdown",
  "knitr",
  "kableExtra"
)

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat("Installing:", pkg, "\n")
    install.packages(pkg, repos = "https://cloud.r-project.org")
  } else {
    cat("Already installed:", pkg, "\n")
  }
}

invisible(lapply(cran_packages, install_if_missing))

if (!requireNamespace("torch", quietly = TRUE)) {
  install.packages("torch")
}
if (!torch::torch_is_installed()) {
  cat("Installing torch backend (LibTorch)...\n")
  torch::install_torch()
}

cat("\nAll packages installed.\n")
cat("Torch backend:", ifelse(torch::torch_is_installed(), "OK", "MISSING"), "\n")
cat("CUDA available:", ifelse(torch::cuda_is_available(), "YES", "NO"), "\n")
