## ══════════════════════════════════════════════════════════════
## Pipeline Explorer — Step-by-Step Data Transformation Demo
##
## Run in PowerShell:
##   Set-Location "C:\Users\mail2\OneDrive\Desktop\R project 1"
##   Rscript -e "shiny::runApp('app_pipeline_viz.R', launch.browser=TRUE)"
## ══════════════════════════════════════════════════════════════

library(shiny)
library(bslib)
library(tidyverse)
library(RNifti)
library(torch)
library(ggplot2)
library(patchwork)
library(viridis)

PROJECT_ROOT <- normalizePath(".", winslash = "/")
setwd(PROJECT_ROOT)

source("R/utils.R")
source("R/01_data_loader.R")
source("R/02_preprocessing.R")
source("R/03_model_3d_unet.R")
source("R/04_region_extractor.R")
source("R/05_region_tracker.R")
source("R/06_feature_extraction.R")
source("R/07_progression_engine.R")
source("R/11_inference.R")

IMAGING_ROOT <- file.path(PROJECT_ROOT, "data", "raw", "LUMIERE", "Imaging")
TARGET_SIZE  <- c(48L, 48L, 48L)

patient_dirs <- sort(list.dirs(IMAGING_ROOT, recursive = FALSE,
                               full.names = FALSE))
patient_dirs <- patient_dirs[grepl("^Patient-", patient_dirs)]

## count timepoints per patient (quick metadata for step 0)
dataset_meta <- tryCatch({
  tibble(pid = patient_dirs) %>%
    rowwise() %>%
    mutate(
      n_tp = length(
        list.dirs(file.path(IMAGING_ROOT, pid),
                  recursive = FALSE, full.names = FALSE)
      )
    ) %>%
    ungroup()
}, error = function(e) tibble(pid = patient_dirs, n_tp = NA_integer_))

message("Loading 3D U-Net model...")
MODEL_PATH <- file.path(PROJECT_ROOT, "models", "unet_3d_trained_jit.pt")
UNET_MODEL <- load_trained_unet(MODEL_PATH, device = "cpu")

## ── Plot helpers ────────────────────────────────────────────

BG    <- "#050a15"
CARD  <- "#0d1f3c"
MUTED <- "#5a7a9a"
TEXT  <- "#c8ddf0"
CYAN  <- "#00d4ff"

SEG_PAL <- c("0" = BG, "1" = "#ef4444", "2" = "#f59e0b",
             "3" = "#22c55e", "4" = "#60a5fa")

void_dark <- function() {
  theme_void() +
    theme(plot.background  = element_rect(fill = BG, colour = NA),
          plot.title       = element_text(hjust = .5, size = 10,
                                          colour = TEXT, face = "bold"))
}

slice_plot <- function(mat, title = "", pal = "inferno") {
  df <- expand.grid(x = seq_len(nrow(mat)), y = seq_len(ncol(mat)))
  df$v <- as.vector(mat)
  ggplot(df, aes(x, y, fill = v)) +
    geom_raster() +
    scale_fill_viridis_c(option = pal, guide = "none") +
    coord_fixed() +
    labs(title = title) +
    void_dark()
}

seg_slice_plot <- function(mat, title = "") {
  df <- expand.grid(x = seq_len(nrow(mat)), y = seq_len(ncol(mat)))
  df$label <- factor(as.vector(mat))
  ggplot(df, aes(x, y, fill = label)) +
    geom_raster() +
    scale_fill_manual(values = SEG_PAL, guide = "none") +
    coord_fixed() +
    labs(title = title) +
    void_dark()
}

format_voxel_patch <- function(volume, label, size = 5L, center = NULL) {
  volume <- as.array(volume)
  dims <- dim(volume)
  if (length(dims) < 3L) return("No 3D voxel data available")

  if (is.null(center)) center <- dims[1:3] %/% 2L
  center <- pmax(1L, pmin(dims[1:3], as.integer(center)))
  half <- size %/% 2L
  x <- seq.int(max(1L, center[1] - half), min(dims[1], center[1] + half))
  y <- seq.int(max(1L, center[2] - half), min(dims[2], center[2] + half))
  patch <- volume[x, y, center[3], drop = TRUE]

  paste0(
    label, "\n",
    "Real axial voxel values; z=", center[3],
    ", x=", min(x), ":", max(x), ", y=", min(y), ":", max(y), "\n",
    paste(capture.output(print(round(patch, 3))), collapse = "\n")
  )
}

format_channel_patches <- function(volume, channel_names) {
  volume <- as.array(volume)
  n_channels <- min(dim(volume)[4], length(channel_names))
  paste(vapply(seq_len(n_channels), function(channel) {
    format_voxel_patch(
      volume[, , , channel],
      paste0(channel_names[channel], " channel"),
      size = 3L
    )
  }, character(1)), collapse = "\n\n")
}

format_table_sample <- function(data, label, rows = 3L) {
  if (is.null(data) || !is.data.frame(data) || nrow(data) == 0L) {
    return(paste0(label, "\nNo rows produced at this stage"))
  }

  paste0(
    label, " (first ", min(rows, nrow(data)), " actual rows)\n",
    paste(capture.output(
      print(as.data.frame(utils::head(data, rows)), row.names = FALSE)
    ), collapse = "\n")
  )
}

## ── CSS ─────────────────────────────────────────────────────

CSS <- paste0("
body{font-family:'Segoe UI',system-ui,sans-serif;}

.step-card{
  background:", CARD, ";border:1px solid rgba(0,212,255,.12);
  border-radius:14px;padding:22px 24px;margin-bottom:6px;
  position:relative;border-left:4px solid #0088ff;
  transition:box-shadow .3s;}
.step-card:hover{box-shadow:0 0 20px rgba(0,212,255,.1);}

.step-badge{
  position:absolute;top:-14px;left:-14px;width:32px;height:32px;
  background:linear-gradient(135deg,#0066cc,#00d4ff);border-radius:50%;
  display:flex;align-items:center;justify-content:center;
  font-weight:800;font-size:14px;color:#fff;
  box-shadow:0 0 14px rgba(0,212,255,.5);}

.step-title{color:#00d4ff;font-size:16px;font-weight:700;margin:0 0 4px;}
.step-desc{color:", MUTED, ";font-size:12px;line-height:1.55;margin:4px 0 12px;}

.data-transform{
  display:grid;grid-template-columns:1fr 36px 1fr;gap:6px;
  align-items:stretch;margin:10px 0;}
.sample-heading{color:#00d4ff;font-size:11px;font-weight:700;
  letter-spacing:.7px;margin:14px 0 6px;}
.dt-box{background:#0a1628;border:1px solid rgba(0,212,255,.08);
  border-radius:8px;padding:10px 12px;font-family:Consolas,monospace;
  font-size:11px;line-height:1.5;color:#7ab8e0;overflow-x:auto;}
.dt-label{font-size:9px;color:#00d4ff;text-transform:uppercase;
  letter-spacing:.8px;margin-bottom:4px;font-weight:700;}
.dt-arrow{display:flex;align-items:center;justify-content:center;
  font-size:26px;color:#0088ff;
  text-shadow:0 0 12px rgba(0,136,255,.6);}

.arrow-down{text-align:center;font-size:28px;color:#0088ff;
  margin:2px 0;text-shadow:0 0 14px rgba(0,136,255,.5);}

.overview-stat{background:#0a1628;border:1px solid rgba(0,212,255,.1);
  border-radius:10px;padding:14px;text-align:center;}
.overview-stat .ov-val{font-size:26px;font-weight:800;color:#00d4ff;
  text-shadow:0 0 12px rgba(0,212,255,.4);}
.overview-stat .ov-lbl{font-size:10px;color:", MUTED, ";
  text-transform:uppercase;letter-spacing:.8px;margin-top:2px;}

.tree{font-family:Consolas,monospace;font-size:12px;color:#7ab8e0;
  background:#0a1628;border:1px solid rgba(0,212,255,.08);
  border-radius:10px;padding:14px;line-height:1.7;white-space:pre;}

.legend-row{display:flex;gap:14px;justify-content:center;margin:8px 0;}
.legend-chip{display:flex;align-items:center;gap:5px;font-size:11px;color:", MUTED, ";}
.legend-dot{width:10px;height:10px;border-radius:3px;display:inline-block;}

.btn-go{background:linear-gradient(135deg,#0066cc,#0088ff);border:none;
  border-radius:10px;font-weight:700;padding:14px 0;font-size:15px;
  width:100%;color:#fff;cursor:pointer;transition:all .15s;}
.btn-go:hover{transform:translateY(-2px);box-shadow:0 0 25px rgba(0,136,255,.5);color:#fff;}

.selectize-input{background:#0a1628!important;color:", TEXT, "!important;
  border:1px solid rgba(0,212,255,.2)!important;border-radius:10px!important;}
.selectize-dropdown{background:", CARD, "!important;color:", TEXT, "!important;
  border:1px solid rgba(0,212,255,.2)!important;}
.selectize-dropdown-content .active{background:#0088ff!important;}

.rano-big{font-size:48px;font-weight:900;text-align:center;margin:10px 0;}

pre{background:#0a1628!important;color:", TEXT, "!important;
  border:1px solid rgba(0,212,255,.08);border-radius:10px;
  padding:12px;font-size:12px;margin:0;}
")

## ════════════════════════════════════════════════════════════
##                            UI
## ════════════════════════════════════════════════════════════

ui <- page_sidebar(
  title = NULL,
  theme = bs_theme(version = 5, bg = BG, fg = TEXT,
                   primary = "#0088ff", font_scale = .95),
  tags$head(tags$style(HTML(CSS))),

  sidebar = sidebar(
    width = 260, bg = "#0a1628", open = TRUE,
    div(style = "padding:8px 4px;",
      div(style = "text-align:center;margin-bottom:10px;",
        tags$span(style = "font-size:24px;", "\U0001F52C"),
        h5("Pipeline Explorer",
           style = "margin:4px 0 0;font-weight:800;color:#fff;"),
        tags$small("Data transformation demo",
                   style = paste0("color:", MUTED, ";font-size:11px;"))
      ),
      hr(style = "border-color:rgba(0,212,255,.15);margin:10px 0;"),
      selectizeInput("patient_id", "PATIENT",
                     choices = patient_dirs,
                     selected = if ("Patient-083" %in% patient_dirs) {
                       "Patient-083"
                     } else if (length(patient_dirs) > 0) {
                       patient_dirs[1]
                     } else {
                       character(0)
                     }),
      sliderInput("slice_z", "Axial Slice",
                  min = 1, max = 182, value = 91, step = 1),
      actionButton("go_btn",
                    "\U0001F680 Start Pipeline",
                    class = "btn btn-primary btn-go",
                    style = "width:100%;"),
      br(), br(),
      htmlOutput("status_box"),
      hr(style = "border-color:rgba(0,212,255,.15);margin:10px 0;"),
      div(style = paste0("font-size:10px;color:", MUTED,
                          ";text-align:center;"),
          "VIT Chennai \U00B7 MTech ISE")
    )
  ),

  ## main area
  uiOutput("main_content")
)

## ════════════════════════════════════════════════════════════
##                         SERVER
## ════════════════════════════════════════════════════════════

server <- function(input, output, session) {

  rv <- reactiveValues(steps = NULL, status = "")

  output$status_box <- renderUI({
    HTML(paste0("<div style='color:", MUTED, ";font-size:12px;'>",
                rv$status, "</div>"))
  })

  ## ── Run pipeline ──────────────────────────────────────────

  observeEvent(input$go_btn, {
    req(input$patient_id)
    pid <- input$patient_id
    pp  <- file.path(IMAGING_ROOT, pid)
    if (!dir.exists(pp)) { rv$status <- "Folder not found"; return() }
    if (is.null(UNET_MODEL)) {
      rv$status <- paste("Trained U-Net model not available:", MODEL_PATH)
      return()
    }

    steps <- list()

    ## ── STEP 1 : Raw MRI ──
    rv$status <- "Step 1/7 — Loading raw MRI..."
    patient_data <- tryCatch(
      load_full_patient(pp),
      error = function(e) {
        rv$status <- paste("Failed to load patient:", conditionMessage(e))
        NULL
      })
    if (is.null(patient_data)) {
      return()
    }
    tp_names  <- names(patient_data$timepoints)
    first_tp  <- patient_data$timepoints[[1]]
    mods      <- first_tp$mri$available
    has_seg   <- !is.null(first_tp$segmentation$mask)
    raw_dims  <- if (length(first_tp$mri$volumes) > 0)
                   dim(first_tp$mri$volumes[[1]]) else NULL

    steps$step1 <- list(
      title = "Load Raw MRI Data",
      desc = paste0(
        "Read NIfTI files (.nii.gz) from disk into R. ",
        "Patient <b>", pid, "</b> has <b>", length(tp_names),
        "</b> timepoints (", paste(tp_names, collapse = ", "),
        ") with <b>", length(mods), "</b> modalities (",
        paste(toupper(mods), collapse = ", "), "). ",
        "Each modality is a 3D array where every cell is one ",
        "voxel — a tiny cube of brain tissue.<br>",
        "<b>T1</b>: brain anatomy. ",
        "<b>T1c</b>: contrast dye highlights active tumor. ",
        "<b>T2</b>: fluid appears bright. ",
        "<b>FLAIR</b>: edema stands out."
      ),
      in_label  = "Input: .nii.gz files on disk",
      in_data   = paste0(
        "Files per timepoint:\n",
        "  T1.nii.gz / CT1.nii.gz / T2.nii.gz / FLAIR.nii.gz\n",
        "  + segmentation/seg_mask.nii.gz\n\n",
        "Format: NIfTI (Neuroimaging Informatics)\n",
        "Compression: gzip"
      ),
      out_label = paste0("Output: 3D arrays [",
                          paste(raw_dims, collapse = "x"), "]"),
      out_data  = paste0(
        "patient_data$timepoints$`", tp_names[1],
        "`$mri$volumes$", mods[1], "\n",
        "  shape: [", paste(raw_dims, collapse = ", "), "]\n",
        "  voxels: ", format(prod(raw_dims), big.mark = ","), "\n",
        "  dtype: float64\n",
        "  intensity range: ",
        round(min(as.array(first_tp$mri$volumes[[1]])), 1), " – ",
        round(max(as.array(first_tp$mri$volumes[[1]])), 1), "\n\n",
        "Modalities loaded: ", length(mods), "\n",
        "Timepoints: ", length(tp_names), "\n",
        "Segmentation: ", if (has_seg) "yes" else "no"
      ),
      in_sample = paste0(
        "Source file read by RNifti::readNifti()\n",
        first_tp$mri$files[[mods[1]]]
      ),
      out_sample = format_voxel_patch(
        as.array(first_tp$mri$volumes[[mods[1]]]),
        paste0(toupper(mods[1]), " loaded into the R array")
      ),
      raw_vols = lapply(first_tp$mri$volumes, as.array),
      seg_vol  = if (has_seg) as.array(first_tp$segmentation$mask) else NULL,
      raw_dims = raw_dims
    )

    ## ── STEP 2 : Preprocessing ──
    rv$status <- "Step 2/7 — Preprocessing..."
    preprocessed <- tryCatch(
      preprocess_full_patient(patient_data, target_size = TARGET_SIZE),
      error = function(e) {
        rv$status <- paste("Preprocessing failed:", conditionMessage(e))
        NULL
      })
    if (is.null(preprocessed)) return()

    pp_shape <- if (!is.null(preprocessed[[1]]$stacked))
                  dim(preprocessed[[1]]$stacked) else NULL
    sample_center <- if (!is.null(first_tp$segmentation$mask)) {
      find_roi_center(as.array(first_tp$segmentation$mask))
    } else {
      NULL
    }

    steps$step2 <- list(
      title = "Preprocess & Normalize",
      desc = paste0(
        "Each raw volume is <b>Z-score normalized</b> (mean=0, std=1) ",
        "so scans from different machines are comparable. A <b>brain mask</b> ",
        "removes skull/air. The volume is <b>cropped to the tumor ROI</b> ",
        "and resized to <b>48\u00D748\u00D748</b>. Finally the 4 modalities ",
        "are <b>stacked</b> into one 4D tensor."
      ),
      in_label  = paste0("Input: Raw [",
                          paste(raw_dims, collapse = "x"),
                          "] per modality"),
      in_data   = paste0(
        "4 separate volumes, each [",
        paste(raw_dims, collapse = ", "), "]\n",
        "Different intensity scales per scanner\n",
        "Not aligned to common space\n",
        "Total voxels: ",
        format(prod(raw_dims) * length(mods), big.mark = ",")
      ),
      out_label = paste0("Output: Stacked [",
                          paste(pp_shape, collapse = "x"), "]"),
      out_data  = paste0(
        "Shape: [", paste(pp_shape, collapse = ", "), "]\n",
        "  dims 1-3: spatial (48 x 48 x 48)\n",
        "  dim 4   : channels (T1, T1c, T2, FLAIR)\n\n",
        "Normalized: mean \u2248 0, std \u2248 1\n",
        "Centered on tumor ROI\n",
        "Voxels: ", format(prod(pp_shape), big.mark = ","), "\n",
        "Reduction: ",
        round(prod(raw_dims) / prod(head(pp_shape, 3)) * 100, 0),
        "x smaller"
      ),
      in_sample = format_voxel_patch(
        as.array(first_tp$mri$volumes[[mods[1]]]),
        paste0("Raw ", toupper(mods[1]), " voxels at preprocessing center"),
        center = sample_center
      ),
      out_sample = format_voxel_patch(
        preprocessed[[1]]$stacked[, , , 1],
        "Normalized/cropped T1 voxels"
      ),
      stacked = preprocessed[[1]]$stacked,
      seg_pp  = preprocessed[[1]]$segmentation
    )

    ## ── STEP 3 : U-Net ──
    rv$status <- "Step 3/7 — Running 3D U-Net..."
    segmented <- tryCatch(
      segment_patient(UNET_MODEL, preprocessed, "cpu",
                      prefer_ground_truth = FALSE),
      error = function(e) {
        rv$status <- paste("U-Net inference failed:", conditionMessage(e))
        NULL
      })
    if (is.null(segmented) || length(segmented) == 0) return()
    first_seg <- segmented[[1]]$segmentation
    labs_found <- sort(unique(as.integer(first_seg)))
    lab_names  <- c("0" = "Background", "1" = "Necrosis",
                    "2" = "Edema", "3" = "Enhancing", "4" = "Enh.Core")

    steps$step3 <- list(
      title = "3D U-Net Prediction",
      desc = paste0(
        "The 4-channel tensor is passed through a <b>3D U-Net</b> ",
        "(22.6M parameters, 32 base filters). The encoder compresses ",
        "spatial information, the decoder reconstructs it with skip ",
        "connections. Each voxel gets 4 class probabilities; ",
        "<b>argmax</b> picks the highest. Expert segmentation labels, ",
        "when present, are shown separately in Step 1 and are not used ",
        "as the U-Net prediction.<br>",
        "<span style='color:#ef4444'>\u25A0 Necrosis</span> (dead core) &nbsp; ",
        "<span style='color:#f59e0b'>\u25A0 Edema</span> (swelling) &nbsp; ",
        "<span style='color:#22c55e'>\u25A0 Enhancing</span> (active tumor)"
      ),
      in_label  = "Input: Tensor [1, 4, 48, 48, 48]",
      in_data   = paste0(
        "Batch: 1\n",
        "Channels: 4 (T1, T1c, T2, FLAIR)\n",
        "Spatial: 48 x 48 x 48\n\n",
        "U-Net architecture:\n",
        "  Encoder: 4\u219216\u219232\u219264\u2192128\n",
        "  Bottleneck: 256 filters\n",
        "  Decoder: 128\u219264\u219232\u219216\u21924\n",
        "  + skip connections at every level\n",
        "  Final: 1\u00D71 conv \u2192 4 class logits"
      ),
      out_label = paste0("Output: Seg mask [",
                          paste(dim(first_seg), collapse = "x"), "]"),
      out_data  = paste0(
        "Shape: [", paste(dim(first_seg), collapse = ", "), "]\n",
        "Labels: [", paste(labs_found, collapse = ", "), "]\n\n",
        paste(sapply(labs_found, function(l) {
          n <- sum(first_seg == l)
          paste0("  ", lab_names[as.character(l)], ": ",
                 format(n, big.mark = ","), " voxels (",
                 round(n / prod(dim(first_seg)) * 100, 1), "%)")
        }), collapse = "\n"), "\n\n",
        "Tumor voxels: ", format(sum(first_seg > 0), big.mark = ","),
        " / ", format(prod(dim(first_seg)), big.mark = ",")
      ),
      in_sample = format_channel_patches(
        preprocessed[[1]]$stacked,
        c("T1", "T1c", "T2", "FLAIR")
      ),
      out_sample = format_voxel_patch(
        first_seg,
        "Predicted segmentation labels (0=background; positive=class)"
      ),
      seg = first_seg
    )

    ## ── STEP 4 : Region Extraction ──
    rv$status <- "Step 4/7 — Extracting regions..."
    region_data <- extract_all_patient_regions(segmented, pid,
                                               min_vol = 50)
    rt <- region_data$table

    steps$step4 <- list(
      title = "Region Extraction",
      desc = paste0(
        "For each tumour label, <b>3D connected-component analysis</b> ",
        "(BFS flood fill, 6-connectivity) finds separate blobs. ",
        "Blobs smaller than 50\u2009mm\u00B3 are discarded. Each surviving ",
        "blob becomes a named region with measured <b>volume</b>, ",
        "<b>centroid</b>, and <b>bounding box</b>."
      ),
      in_label = "Input: Seg mask per timepoint",
      in_data  = paste0(
        "Segmentation masks: ", length(segmented), " timepoints\n",
        "Labels: 1 = Necrosis, 2 = Edema, 3 = Enhancing\n",
        "Algorithm: 3D BFS (6-connectivity)\n",
        "Min volume: 50 mm\u00B3"
      ),
      out_label = paste0("Output: ", nrow(rt), " regions"),
      out_data  = paste0(
        "Total regions: ", nrow(rt), "\n\n",
        paste(rt %>%
                group_by(timepoint) %>%
                summarise(n = n(),
                          vol = round(sum(volume_cm3), 1),
                          .groups = "drop") %>%
                mutate(s = paste0("  ", timepoint, ": ", n,
                                  " regions, ", vol, " cm\u00B3")) %>%
                pull(s),
              collapse = "\n"), "\n\n",
        "Region types:\n",
        paste(rt %>% count(region_type) %>%
                mutate(s = paste0("  ", region_type, ": ", n)) %>%
                pull(s),
              collapse = "\n")
      ),
      in_sample = format_voxel_patch(
        first_seg,
        "Predicted voxel labels passed to region extraction"
      ),
      out_sample = format_table_sample(
        rt,
        "Extracted regions (patient/timepoint/type/volume/centroid)"
      ),
      region_table = rt,
      seg = first_seg
    )

    ## ── STEP 5 : Tracking ──
    rv$status <- "Step 5/7 — Tracking regions..."
    events <- NULL
    if (length(region_data$regions_by_timepoint) >= 2) {
      tracking <- track_patient_regions(
        region_data$regions_by_timepoint)
      events <- create_event_table(tracking$all_events, pid)
    }
    n_ev <- if (!is.null(events)) nrow(events) else 0L

    steps$step5 <- list(
      title = "Longitudinal Region Tracking",
      desc = paste0(
        "Regions from consecutive timepoints are matched using a ",
        "<b>combined score</b>: 50% spatial overlap (IoU), 30% centroid ",
        "proximity, 20% volume similarity. Threshold = 0.15. Unmatched ",
        "regions become <b>appearances</b> or <b>disappearances</b>; ",
        "one-to-many = <b>split</b>; many-to-one = <b>merge</b>."
      ),
      in_label = "Input: Regions from consecutive timepoints",
      in_data  = paste0(
        "Matching score:\n",
        "  0.5 \u00D7 IoU (voxel overlap)\n",
        "  0.3 \u00D7 exp(-dist / 20) (centroid)\n",
        "  0.2 \u00D7 (1 - |volA-volB| / max)\n",
        "  + 0.1 type bonus\n\n",
        "Greedy assignment (highest first)\n",
        "Threshold: 0.15"
      ),
      out_label = paste0("Output: ", n_ev, " events"),
      out_data  = if (n_ev > 0) {
        paste0(
          "Events:\n",
          paste(events %>% count(event_type) %>%
                  mutate(s = paste0("  ", event_type, ": ", n)) %>%
                  pull(s), collapse = "\n")
        )
      } else "No events (need 2+ timepoints with regions)",
      in_sample = format_table_sample(
        rt,
        "Real region rows being matched"
      ),
      out_sample = format_table_sample(
        events,
        "Actual tracking events"
      ),
      events = events
    )

    ## ── STEP 6 : Feature Extraction ──
    rv$status <- "Step 6/7 — Extracting features..."
    long_feat <- if (n_ev > 0 && nrow(rt) > 0)
      compute_longitudinal_features(rt, events) else tibble()
    pat_sum   <- if (nrow(rt) > 0 && n_ev > 0)
      compute_patient_summary_features(
        rt, events, long_feat) else tibble()

    steps$step6 <- list(
      title = "Feature Extraction",
      desc = paste0(
        "For every pair of matched regions (continuation events), ",
        "compute <b>absolute & percentage volume change</b>, ",
        "<b>centroid displacement</b> (mm), and aggregate per-patient ",
        "summaries: total volume first/last, # growing/shrinking regions, ",
        "max growth rate, event counts."
      ),
      in_label = "Input: Region pairs across time",
      in_data  = paste0(
        "Continuation pairs: ",
        if (n_ev > 0) sum(events$event_type == "CONTINUATION") else 0,
        "\n",
        "Per pair:\n",
        "  vol_t1, vol_t2 \u2192 abs_change, pct_change\n",
        "  centroid_t1, centroid_t2 \u2192 displacement (mm)"
      ),
      out_label = paste0("Output: ",
                          nrow(long_feat), " feature rows + ",
                          nrow(pat_sum), " patient summary"),
      out_data  = if (nrow(pat_sum) > 0) {
        ps <- pat_sum[1, ]
        paste0(
          "Patient summary:\n",
          "  total_vol_first: ", round(ps$total_vol_first, 2), " cm\u00B3\n",
          "  total_vol_last:  ", round(ps$total_vol_last, 2), " cm\u00B3\n",
          "  pct_change:      ",
          sprintf("%+.1f%%", ps$total_vol_pct_change * 100), "\n",
          "  growing regions: ", ps$n_growing_regions, "\n",
          "  shrinking:       ", ps$n_shrinking_regions, "\n",
          "  total events:    ", ps$total_events
        )
      } else "Insufficient data",
      in_sample = paste(
        format_table_sample(rt, "Actual region measurement rows"),
        format_table_sample(events, "Actual matched/event rows"),
        sep = "\n\n"
      ),
      out_sample = paste(
        format_table_sample(long_feat, "Computed longitudinal feature rows"),
        format_table_sample(pat_sum, "Computed patient summary row"),
        sep = "\n\n"
      ),
      long_feat = long_feat,
      pat_sum   = pat_sum
    )

    ## ── STEP 7 : Classification ──
    rv$status <- "Step 7/7 — Classifying progression..."
    progression  <- NULL
    trajectories <- NULL

    if (length(region_data$regions_by_timepoint) >= 2 &&
        !is.null(events)) {
      tp_r <- names(region_data$regions_by_timepoint)
      tw   <- sapply(patient_data$timepoints[tp_r], `[[`,
                     "week_number")
      trajectories <- list()
      for (r in region_data$regions_by_timepoint[[1]]) {
        vols <- sapply(tp_r, function(tp) {
          m <- Filter(function(x) x$region_type == r$region_type,
                      region_data$regions_by_timepoint[[tp]])
          if (length(m) > 0) m[[1]]$volume_cm3 else 0
        })
        trajectories[[r$region_id]] <-
          analyze_region_trajectory(vols, tw)
      }
      progression <- classify_patient_progression(trajectories,
                                                   events)
    }

    cls  <- if (!is.null(progression)) progression$classification else "N/A"
    conf <- if (!is.null(progression))
              round(progression$confidence * 100) else 0

    steps$step7 <- list(
      title = "RANO Progression Classification",
      desc = paste0(
        "Each region's trajectory is tagged GROWTH / SHRINKAGE / ",
        "STABLE / DISAPPEARED. These are combined with events to ",
        "produce a <b>RANO classification</b>:<br>",
        "<b>CR</b> = all gone &nbsp; ",
        "<b>PR</b> = shrinking &nbsp; ",
        "<b>SD</b> = stable &nbsp; ",
        "<b>PD</b> = growing / new lesions.<br>",
        "Region-wise analysis catches <b>hidden changes</b> that ",
        "total tumour volume alone would miss."
      ),
      in_label = "Input: Trajectories + Events",
      in_data  = if (!is.null(trajectories)) {
        paste0(
          "Region trajectories:\n",
          paste(sapply(names(trajectories), function(rid) {
            t <- trajectories[[rid]]
            paste0("  ", rid, ": ", t$trend, "  ",
                   round(t$first_volume, 1), "\u2192",
                   round(t$last_volume, 1), " cm\u00B3")
          }), collapse = "\n"), "\n\n",
          "Decision rules:\n",
          "  all gone        \u2192 CR\n",
          "  all shrinking   \u2192 PR\n",
          "  stable + no new \u2192 SD\n",
          "  growth/new/split\u2192 PD"
        )
      } else "Insufficient data",
      out_label = paste0("Output: ", cls, " (", conf, "%)"),
      out_data  = if (!is.null(progression)) {
        ev <- progression$evidence
        paste0(
          "Classification: ", cls, "\n",
          "Confidence:     ", conf, "%\n\n",
          "Evidence:\n",
          "  Growing:      ", ev$n_growing, "\n",
          "  Shrinking:    ", ev$n_shrinking, "\n",
          "  Stable:       ", ev$n_stable, "\n",
          "  Appeared:     ", ev$n_appeared, "\n",
          "  Disappeared:  ", ev$n_disappeared, "\n",
          "  Splits:       ", ev$has_splits, "\n",
          "  New regions:  ", ev$has_new_regions
        )
      } else "Not enough data to classify",
      in_sample = paste(
        if (!is.null(trajectories) && length(trajectories) > 0) {
          paste0(
            "Actual per-region trajectories\n",
            paste(vapply(names(trajectories), function(region_id) {
              trajectory <- trajectories[[region_id]]
              paste0(
                region_id, ": trend=", trajectory$trend,
                ", volume=", round(trajectory$first_volume, 3),
                " -> ", round(trajectory$last_volume, 3), " cm3"
              )
            }, character(1)), collapse = "\n")
          )
        } else {
          format_table_sample(rt, "Actual extracted region rows")
        },
        format_table_sample(events, "Actual events used by classification"),
        sep = "\n\n"
      ),
      out_sample = if (!is.null(progression)) {
        ev <- progression$evidence
        paste0(
          "Actual classification result\n",
          "classification = ", progression$classification, "\n",
          "confidence = ", round(progression$confidence, 3), "\n",
          "evidence = { growing: ", ev$n_growing,
          ", shrinking: ", ev$n_shrinking,
          ", stable: ", ev$n_stable,
          ", appeared: ", ev$n_appeared,
          ", disappeared: ", ev$n_disappeared,
          ", splits: ", ev$has_splits,
          ", new_regions: ", ev$has_new_regions, " }"
        )
      } else {
        "No classification result: insufficient trajectory/event data"
      },
      cls = cls, conf = conf,
      progression = progression,
      trajectories = trajectories,
      region_table = rt
    )

    rv$steps  <- steps
    rv$status <- paste0("\u2705 Pipeline complete — ", cls,
                        " (", conf, "%)")
  })

  ## ── Render main area ──────────────────────────────────────

  output$main_content <- renderUI({
    ## always show dataset overview
    overview <- div(
      div(class = "step-card",
          style = "border-left-color:#00d4ff;",
        div(class = "step-badge", "0"),
        div(class = "step-title", "Dataset Overview — LUMIERE"),
        div(class = "step-desc", HTML(paste0(
          "The <b>LUMIERE</b> dataset contains longitudinal brain MRI scans ",
          "of glioma patients. Each patient has multiple timepoints — ",
          "MRI scans taken weeks or months apart to monitor tumour ",
          "progression. Each scan includes <b>4 MRI modalities</b> ",
          "(T1, T1c, T2, FLAIR) and expert <b>tumour segmentation</b>."
        ))),
        layout_columns(
          col_widths = c(3, 3, 3, 3), fill = FALSE,
          div(class = "overview-stat",
              div(class = "ov-val", length(patient_dirs)),
              div(class = "ov-lbl", "Patients")),
          div(class = "overview-stat",
              div(class = "ov-val",
                  sum(dataset_meta$n_tp, na.rm = TRUE)),
              div(class = "ov-lbl", "Total Timepoints")),
          div(class = "overview-stat",
              div(class = "ov-val", "4"),
              div(class = "ov-lbl", "MRI Modalities")),
          div(class = "overview-stat",
              div(class = "ov-val", ".nii.gz"),
              div(class = "ov-lbl", "File Format"))
        ),
        br(),
        div(class = "tree", HTML(paste0(
          "LUMIERE / Imaging /\n",
          "  \u251C\u2500 Patient-001 /\n",
          "  \u2502   \u251C\u2500 week-000 /\n",
          "  \u2502   \u2502   \u251C\u2500 T1.nii.gz        ",
          "    <span style='color:#22c55e'>brain anatomy</span>\n",
          "  \u2502   \u2502   \u251C\u2500 CT1.nii.gz       ",
          "    <span style='color:#ef4444'>contrast (active tumour)</span>\n",
          "  \u2502   \u2502   \u251C\u2500 T2.nii.gz        ",
          "    <span style='color:#f59e0b'>fluid / edema</span>\n",
          "  \u2502   \u2502   \u251C\u2500 FLAIR.nii.gz     ",
          "    <span style='color:#60a5fa'>edema without CSF</span>\n",
          "  \u2502   \u2502   \u2514\u2500 segmentation /   ",
          "    <span style='color:#a855f7'>expert tumour labels</span>\n",
          "  \u2502   \u2514\u2500 week-044 / \u2026\n",
          "  \u251C\u2500 Patient-002 / \u2026\n",
          "  \u2514\u2500 Patient-091 / \u2026"
        )))
      ),
      div(class = "arrow-down", "\u2B07")
    )

    if (is.null(rv$steps)) {
      return(tagList(
        overview,
        div(style = paste0("text-align:center;padding:40px;color:",
                            MUTED, ";"),
            h4("Select a patient and click Start Pipeline",
               style = "font-weight:300;"),
            p("Each step will show the data transformation."))
      ))
    }

    ## Build step cards
    steps <- rv$steps
    z <- input$slice_z

    step_tags <- lapply(seq_along(steps), function(i) {
      s  <- steps[[i]]
      nm <- names(steps)[i]

      tagList(
        div(class = "step-card",
          div(class = "step-badge", i),
          div(class = "step-title", s$title),
          div(class = "step-desc", HTML(s$desc)),
          div(class = "data-transform",
            div(class = "dt-box",
                div(class = "dt-label", s$in_label),
                tags$pre(s$in_data)),
            div(class = "dt-arrow", "\u27A1"),
            div(class = "dt-box",
                div(class = "dt-label", s$out_label),
                tags$pre(s$out_data))
          ),
          div(class = "sample-heading", "ACTUAL DATA — SMALL INPUT/OUTPUT SAMPLE"),
          div(class = "data-transform",
            div(class = "dt-box",
                div(class = "dt-label", "Real input values"),
                tags$pre(s$in_sample)),
            div(class = "dt-arrow", "\u27A1"),
            div(class = "dt-box",
                div(class = "dt-label", "Real output values"),
                tags$pre(s$out_sample))
          ),
          plotOutput(paste0("plot_", i), height = "220px")
        ),
        if (i < length(steps))
          div(class = "arrow-down", "\u2B07")
        else
          div(class = "arrow-down", "\u2705")
      )
    })

    do.call(tagList, c(list(overview), step_tags))
  })

  ## ── Dynamic plots ─────────────────────────────────────────

  observe({
    req(rv$steps)
    z <- input$slice_z
    steps <- rv$steps

    for (idx in seq_along(steps)) {
      local({
        i  <- idx
        s  <- steps[[i]]
        nm <- names(steps)[i]

        output[[paste0("plot_", i)]] <- renderPlot({
          ## Step 1: raw modalities
          if (nm == "step1" && !is.null(s$raw_vols)) {
            plots <- lapply(names(s$raw_vols), function(mod) {
              vol <- s$raw_vols[[mod]]
              zz  <- min(z, dim(vol)[3])
              slice_plot(vol[, , zz], toupper(mod))
            })
            if (!is.null(s$seg_vol)) {
              sv <- s$seg_vol
              zz <- min(z, dim(sv)[3])
              plots <- c(plots, list(
                seg_slice_plot(sv[, , zz], "Ground Truth")))
            }
            wrap_plots(plots, nrow = 1) +
              plot_annotation(
                theme = theme(
                  plot.background = element_rect(fill = BG,
                                                  colour = NA)))
          }

          ## Step 2: before / after
          else if (nm == "step2") {
            plist <- list()
            if (!is.null(s$stacked)) {
              stk <- s$stacked
              zz  <- min(z, dim(stk)[3])
              mods <- c("T1", "T1c", "T2", "FLAIR")
              for (ch in seq_len(min(4, dim(stk)[4]))) {
                plist <- c(plist, list(
                  slice_plot(stk[, , zz, ch],
                             paste0(mods[ch], " 48\u00B3"))))
              }
            }
            if (length(plist) == 0) return(NULL)
            wrap_plots(plist, nrow = 1) +
              plot_annotation(
                title = "Preprocessed (normalized, 48\u00D748\u00D748)",
                theme = theme(
                  plot.title = element_text(
                    colour = TEXT, face = "bold", size = 12),
                  plot.background = element_rect(
                    fill = BG, colour = NA)))
          }

          ## Step 3: seg mask
          else if (nm == "step3" && !is.null(s$seg)) {
            seg <- s$seg
            zz  <- min(z, dim(seg)[3])
            seg_slice_plot(seg[, , zz], "U-Net Prediction")
          }

          ## Step 4: region table
          else if (nm == "step4" && !is.null(s$region_table) &&
                   nrow(s$region_table) > 0) {
            pal <- c(necrosis = "#ef4444", edema = "#f59e0b",
                     enhancing = "#22c55e", label_4 = "#60a5fa")
            ggplot(s$region_table,
                   aes(timepoint, volume_cm3,
                       fill = region_type)) +
              geom_col(position = "dodge", alpha = .85,
                       width = .6) +
              scale_fill_manual(values = pal, name = "Region") +
              labs(title = "Region Volumes by Timepoint",
                   x = "Timepoint", y = "Volume (cm\u00B3)") +
              theme_minimal(base_size = 11) +
              theme(
                plot.background = element_rect(fill = BG,
                                                colour = NA),
                panel.background = element_rect(fill = BG,
                                                 colour = NA),
                text = element_text(colour = TEXT),
                axis.text = element_text(colour = MUTED),
                panel.grid.major = element_line(
                  colour = "#1a2a4a", linewidth = .3),
                panel.grid.minor = element_blank(),
                legend.background = element_rect(fill = BG,
                                                  colour = NA),
                legend.text = element_text(colour = TEXT),
                plot.title = element_text(face = "bold",
                                          colour = "#fff"))
          }

          ## Step 5: event bar
          else if (nm == "step5" && !is.null(s$events) &&
                   nrow(s$events) > 0) {
            epal <- c(CONTINUATION = "#22c55e",
                      APPEARANCE   = "#ef4444",
                      DISAPPEARANCE = "#a855f7",
                      SPLIT = "#f59e0b", MERGE = "#60a5fa")
            s$events %>%
              count(event_type) %>%
              ggplot(aes(event_type, n, fill = event_type)) +
              geom_col(alpha = .85, width = .6) +
              geom_text(aes(label = n), vjust = -.5,
                        colour = "#fff", fontface = "bold") +
              scale_fill_manual(values = epal, guide = "none") +
              labs(title = "Tracked Events", x = "", y = "Count") +
              theme_minimal(base_size = 11) +
              theme(
                plot.background = element_rect(fill = BG,
                                                colour = NA),
                panel.background = element_rect(fill = BG,
                                                 colour = NA),
                text = element_text(colour = TEXT),
                axis.text = element_text(colour = MUTED),
                panel.grid.major = element_line(
                  colour = "#1a2a4a", linewidth = .3),
                panel.grid.minor = element_blank(),
                plot.title = element_text(face = "bold",
                                          colour = "#fff"))
          }

          ## Step 6: volume change
          else if (nm == "step6" && !is.null(s$long_feat) &&
                   nrow(s$long_feat) > 0) {
            ggplot(s$long_feat,
                   aes(region_type, pct_volume_change,
                       fill = region_type)) +
              geom_col(alpha = .85, width = .5) +
              geom_hline(yintercept = c(-.25, .25),
                         linetype = "dashed",
                         colour = MUTED) +
              scale_fill_manual(
                values = c(necrosis = "#ef4444",
                           edema = "#f59e0b",
                           enhancing = "#22c55e"),
                guide = "none") +
              labs(title = "Volume Change per Region",
                   x = "", y = "% Change") +
              theme_minimal(base_size = 11) +
              theme(
                plot.background = element_rect(fill = BG,
                                                colour = NA),
                panel.background = element_rect(fill = BG,
                                                 colour = NA),
                text = element_text(colour = TEXT),
                axis.text = element_text(colour = MUTED),
                panel.grid.major = element_line(
                  colour = "#1a2a4a", linewidth = .3),
                panel.grid.minor = element_blank(),
                plot.title = element_text(face = "bold",
                                          colour = "#fff"))
          }

          ## Step 7: RANO result
          else if (nm == "step7") {
            cls   <- s$cls
            col <- switch(cls, CR = "#22c55e", PR = "#60a5fa",
                           SD = "#f59e0b", PD = "#ef4444",
                           MUTED)
            label <- switch(cls,
              CR = "Complete Response",
              PR = "Partial Response",
              SD = "Stable Disease",
              PD = "Progressive Disease", "Unknown")
            ggplot() +
              annotate("text", x = .5, y = .65, label = cls,
                       size = 28, fontface = "bold", colour = col) +
              annotate("text", x = .5, y = .35, label = label,
                       size = 7, colour = MUTED) +
              annotate("text", x = .5, y = .18,
                       label = paste0("Confidence: ",
                                      s$conf, "%"),
                       size = 5, colour = MUTED) +
              xlim(0, 1) + ylim(0, 1) +
              theme_void() +
              theme(plot.background = element_rect(fill = BG,
                                                    colour = NA))
          }

          else NULL
        }, bg = BG, height = 220)
      })
    }
  })
}

shinyApp(ui, server)
