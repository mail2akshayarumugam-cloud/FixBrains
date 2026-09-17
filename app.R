# ══════════════════════════════════════════════════════════════
# Tumor Progression Analysis — Shiny Dashboard
# Run: Rscript -e "shiny::runApp('app.R', launch.browser=TRUE)"
# ══════════════════════════════════════════════════════════════

library(shiny)
library(bslib)
library(tidyverse)
library(RNifti)
library(torch)
library(ggplot2)
library(patchwork)
library(viridis)
library(DT)

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
MODEL_PATH   <- file.path(PROJECT_ROOT, "models", "unet_3d_trained_jit.pt")
FIGURES_PATH  <- file.path(PROJECT_ROOT, "results", "figures")
TABLES_PATH   <- file.path(PROJECT_ROOT, "results", "tables")
TARGET_SIZE   <- c(48L, 48L, 48L)

patient_list <- sort(list.dirs(IMAGING_ROOT, recursive = FALSE, full.names = FALSE))
patient_list <- patient_list[grepl("^Patient-", patient_list)]

existing_figs <- if (dir.exists(FIGURES_PATH)) {
  sort(list.files(FIGURES_PATH, pattern = "\\.png$", full.names = FALSE))
} else character(0)

existing_tables <- if (dir.exists(TABLES_PATH)) {
  sort(list.files(TABLES_PATH, pattern = "\\.csv$", full.names = FALSE))
} else character(0)

# ── Shared ggplot dark theme ─────────────────────────────────

dark_theme <- function(base_size = 13) {
  theme_minimal(base_size = base_size) %+replace%
    theme(
      plot.background   = element_rect(fill = "#1c1014", color = NA),
      panel.background  = element_rect(fill = "#1c1014", color = NA),
      text              = element_text(color = "#e0d0d5"),
      axis.text         = element_text(color = "#a0888e"),
      axis.title        = element_text(color = "#c0a8ae"),
      panel.grid.major  = element_line(color = "#2e1a20", linewidth = 0.3),
      panel.grid.minor  = element_blank(),
      legend.background = element_rect(fill = "#1c1014", color = NA),
      legend.text       = element_text(color = "#e0d0d5"),
      legend.title      = element_text(color = "#c0a8ae"),
      strip.text        = element_text(color = "#e0d0d5", face = "bold"),
      plot.title        = element_text(face = "bold", color = "#ffffff", size = rel(1.15)),
      plot.subtitle     = element_text(color = "#a0888e", size = rel(0.9))
    )
}

REGION_PAL <- c(necrosis = "#ef4444", edema = "#f59e0b", enhancing = "#22c55e", label_4 = "#60a5fa")
SEG_PAL    <- c("0" = "#1c1014", "1" = "#ef4444", "2" = "#f59e0b", "3" = "#22c55e", "4" = "#60a5fa")
SEG_LABS   <- c("0" = "Background", "1" = "Necrosis", "2" = "Edema", "3" = "Enhancing", "4" = "Enh. Core")
EVENT_PAL  <- c(CONTINUATION = "#22c55e", APPEARANCE = "#ef4444", DISAPPEARANCE = "#a855f7",
                SPLIT = "#f59e0b", MERGE = "#60a5fa")
RANO_PAL   <- c(CR = "#22c55e", PR = "#60a5fa", SD = "#f59e0b", PD = "#ef4444")

# ── Custom CSS ───────────────────────────────────────────────

custom_css <- "
  /* global */
  body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; }

  /* sidebar */
  .sidebar-panel { padding: 0 !important; }

  .run-btn {
    background: linear-gradient(135deg, #8b1a2b, #a0283c);
    border: none; border-radius: 10px; font-weight: 700;
    padding: 14px 0; font-size: 15px; width: 100%;
    letter-spacing: 0.3px; transition: transform 0.15s, box-shadow 0.15s;
    color: #fff; cursor: pointer;
  }
  .run-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(139,26,43,0.55);
    color: #fff; background: linear-gradient(135deg, #a0283c, #c0354a);
  }

  /* stat cards */
  .stat-card {
    background: #2a1018; border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px; padding: 18px 14px; text-align: center;
    transition: transform 0.15s; border-left: 3px solid #8b1a2b;
  }
  .stat-card:hover { transform: translateY(-2px); box-shadow: 0 4px 16px rgba(139,26,43,0.3); }
  .stat-icon { font-size: 22px; margin-bottom: 6px; }
  .stat-value { font-size: 30px; font-weight: 800; line-height: 1.1; }
  .stat-label { font-size: 11px; color: #a0888e; margin-top: 4px; text-transform: uppercase;
                letter-spacing: 0.8px; }

  /* progress colours */
  .clr-cr { color: #22c55e; } .clr-pr { color: #60a5fa; }
  .clr-sd { color: #f59e0b; } .clr-pd { color: #ef4444; }

  /* cards */
  .dash-card {
    background: #2a1018; border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px; padding: 20px; margin-bottom: 16px;
  }
  .dash-card h4 { margin-top: 0; color: #e0d0d5; font-size: 15px; font-weight: 700; }

  /* patient description banner */
  .patient-banner {
    background: linear-gradient(135deg, #3a0f1a, #2a1018);
    border: 1px solid #5a2030; border-radius: 14px;
    padding: 20px 24px; margin-bottom: 18px;
  }
  .patient-banner h3 { margin: 0 0 12px 0; color: #fff; font-size: 20px; }
  .patient-banner .pb-grid { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 16px; }
  .patient-banner .pb-item { }
  .patient-banner .pb-label { font-size: 11px; color: #a0888e; text-transform: uppercase;
                              letter-spacing: 0.6px; }
  .patient-banner .pb-value { font-size: 15px; color: #e0d0d5; font-weight: 600; margin-top: 2px; }

  /* selectize dark */
  .selectize-input {
    background: #1c1014 !important; color: #e0d0d5 !important;
    border: 1px solid #5a2030 !important; border-radius: 10px !important;
    padding: 10px 12px !important;
  }
  .selectize-dropdown { background: #2a1018 !important; color: #e0d0d5 !important;
                        border: 1px solid #5a2030 !important; border-radius: 10px !important; }
  .selectize-dropdown-content .option { color: #e0d0d5 !important; padding: 8px 12px !important; }
  .selectize-dropdown-content .active { background: #8b1a2b !important; }

  /* text input dark */
  .form-control { background: #1c1014 !important; color: #e0d0d5 !important;
                  border: 1px solid #5a2030 !important; border-radius: 10px !important; }

  /* radio buttons */
  .radio-inline label, .radio label { color: #c0a8ae !important; }

  /* nav pills override */
  .nav-pills .nav-link { color: #a0888e; border-radius: 10px; padding: 8px 18px;
                          font-weight: 600; font-size: 13px; }
  .nav-pills .nav-link.active { background: #8b1a2b; color: #fff; }

  /* data-table dark */
  table.dataTable { color: #e0d0d5 !important; }
  table.dataTable thead th { color: #a0888e !important; border-bottom: 1px solid #5a2030 !important; }
  table.dataTable tbody td { border-top: 1px solid #2a1018 !important; }
  table.dataTable tbody tr { background: #2a1018 !important; }
  table.dataTable tbody tr:hover { background: #3a1525 !important; }
  .dataTables_wrapper .dataTables_filter input,
  .dataTables_wrapper .dataTables_length select {
    background: #1c1014; color: #e0d0d5; border: 1px solid #5a2030; border-radius: 6px; padding: 4px 8px;
  }
  .dataTables_wrapper .dataTables_info,
  .dataTables_wrapper .dataTables_paginate .paginate_button { color: #a0888e !important; }

  /* welcome screen */
  .welcome-screen { text-align: center; padding: 80px 40px; }
  .welcome-screen h3 { color: #8b6070; font-weight: 300; font-size: 22px; }
  .welcome-screen p  { color: #6b4050; }

  /* status badge */
  .status-msg { font-size: 13px; color: #a0888e; min-height: 18px; padding: 4px 0; }

  /* scrollable image gallery */
  .fig-gallery { max-height: 600px; overflow-y: auto; padding-right: 8px; }
  .fig-gallery img { width: 100%; border-radius: 10px; margin-bottom: 14px;
                     border: 1px solid rgba(255,255,255,0.06); }

  /* overlay legend strip */
  .legend-strip { display: flex; gap: 16px; justify-content: center; padding: 8px 0; }
  .legend-chip { display: flex; align-items: center; gap: 5px; font-size: 12px; color: #c0a8ae; }
  .legend-dot  { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }

  /* patient info box */
  .info-row { display: flex; justify-content: space-between; padding: 3px 0;
              font-size: 13px; border-bottom: 1px solid rgba(255,255,255,0.04); }
  .info-label { color: #a0888e; }
  .info-value { color: #e0d0d5; font-weight: 600; }

  /* report pre */
  pre { background: #1c1014 !important; color: #e0d0d5 !important; border: 1px solid #3a1525;
        border-radius: 10px; padding: 16px; font-size: 13px; line-height: 1.6; }
"

# ── UI ────────────────────────────────────────────────────────

ui <- page_sidebar(
  title = NULL,
  theme = bs_theme(
    version = 5,
    bg = "#140a0e", fg = "#e0d0d5",
    primary = "#8b1a2b", secondary = "#2a1018",
    font_scale = 0.95,
    "card-bg" = "#2a1018"
  ),
  tags$head(tags$style(HTML(custom_css))),

  sidebar = sidebar(
    width = 300, bg = "#2a1018", open = TRUE,

    div(style = "padding: 8px 4px 0;",
      div(style = "text-align:center; margin-bottom: 14px;",
        tags$span(style = "font-size:28px;", "\U0001F9E0"),
        h5("Tumor Progression", style = "margin:4px 0 0; font-weight:800; color:#fff;"),
        tags$small("Region-Wise Longitudinal Analysis",
                   style = "color:#94a3b8; font-size:11px;")
      ),
      hr(style = "border-color:#334155; margin: 10px 0;"),

      radioButtons("input_mode", NULL,
                    choices = c("LUMIERE Patient" = "lumiere", "Custom Path" = "custom"),
                    selected = "lumiere", inline = TRUE),
      conditionalPanel(
        condition = "input.input_mode == 'lumiere'",
        selectizeInput("patient_id", "Patient",
                       choices = patient_list, selected = "Patient-083",
                       options = list(placeholder = "Search patient..."))
      ),
      conditionalPanel(
        condition = "input.input_mode == 'custom'",
        textInput("custom_path", "Patient Folder Path",
                  placeholder = "C:/path/to/Patient-XXX"),
        tags$small(style = "color:#64748b; font-size:11px;",
          "Folder must contain week-XXX/ subfolders with .nii.gz MRI files")
      ),

      actionButton("run_btn", "Analyze Patient", class = "run-btn"),
      br(), br(),
      htmlOutput("status_text"),

      hr(style = "border-color:#334155; margin: 12px 0;"),
      htmlOutput("patient_info"),

      hr(style = "border-color:#334155; margin: 12px 0;"),
      div(style = "font-size:11px; color:#64748b; text-align:center; padding-top:4px;",
        "VIT Chennai &middot; MTech ISE"
      )
    )
  ),

  # ── Main content ──

  # Patient description banner (shows after analysis)
  conditionalPanel(
    condition = "output.has_results",
    htmlOutput("patient_banner")
  ),

  layout_columns(
    col_widths = c(3, 3, 3, 3),
    fill = FALSE,
    div(class = "stat-card", htmlOutput("stat_progression")),
    div(class = "stat-card", htmlOutput("stat_regions")),
    div(class = "stat-card", htmlOutput("stat_events")),
    div(class = "stat-card", htmlOutput("stat_confidence"))
  ),

  navset_pill(
    id = "main_tabs",

    # ---- Tab 0: Patient Summary ----
    nav_panel(
      title = "Patient Summary",
      icon  = icon("user-injured"),
      br(),
      htmlOutput("patient_clinical_summary")
    ),

    # ---- Tab 1: MRI Modalities ----
    nav_panel(
      title = "MRI Scans",
      icon  = icon("brain"),
      br(),
      div(class = "dash-card",
        h4("Multi-Modal MRI — Axial Slice"),
        tags$p(style = "color:#a0888e; font-size:13px; margin-bottom:12px;",
          "Each column shows a different MRI modality for the same brain slice. ",
          tags$b("T1"), " shows brain anatomy. ",
          tags$b("T1c (CT1)"), " highlights active tumor with contrast dye. ",
          tags$b("T2"), " makes fluid bright — edema shows up clearly. ",
          tags$b("FLAIR"), " suppresses fluid to isolate tumor swelling from normal CSF. ",
          "Rows show different timepoints — compare across time to see tumor changes."
        ),
        plotOutput("mri_all_modalities", height = "520px")
      )
    ),

    # ---- Tab 2: Segmentation ----
    nav_panel(
      title = "Segmentation",
      icon  = icon("crosshairs"),
      br(),
      div(style = "color:#a0888e; font-size:13px; margin-bottom:12px; padding: 0 4px;",
        "The 3D U-Net model (5.6M parameters, trained on 599 samples) predicts tumor regions for every voxel. ",
        tags$b("Left:"), " Raw segmentation mask — each color is a different tumor sub-region. ",
        tags$b("Right:"), " Segmentation overlaid on the FLAIR MRI so you can see exactly where in the brain the tumor regions are located."
      ),
      layout_columns(
        col_widths = c(6, 6),
        div(class = "dash-card",
          h4("Segmentation Mask"),
          plotOutput("seg_plot", height = "420px")
        ),
        div(class = "dash-card",
          h4("Overlay on FLAIR"),
          plotOutput("overlay_plot", height = "420px")
        )
      ),
      div(class = "legend-strip",
        div(class = "legend-chip", div(class = "legend-dot", style = "background:#ef4444;"), "Necrosis (dead core)"),
        div(class = "legend-chip", div(class = "legend-dot", style = "background:#f59e0b;"), "Edema (swelling)"),
        div(class = "legend-chip", div(class = "legend-dot", style = "background:#22c55e;"), "Enhancing (active tumor)"),
        div(class = "legend-chip", div(class = "legend-dot", style = "background:#60a5fa;"), "Enh. Core")
      )
    ),

    # ---- Tab 3: Regions ----
    nav_panel(
      title = "Regions",
      icon  = icon("chart-bar"),
      br(),
      div(style = "color:#a0888e; font-size:13px; margin-bottom:12px; padding: 0 4px;",
        "Each connected blob of tumor voxels is extracted as a separate region. ",
        tags$b("Bar chart:"), " Volume of each region type at each timepoint — compare heights to see growth or shrinkage. ",
        tags$b("Box plot:"), " Distribution of region volumes — shows typical size and outliers. ",
        "The table below lists every region with its exact volume, location (centroid), and bounding box."
      ),
      layout_columns(
        col_widths = c(7, 5),
        div(class = "dash-card",
          h4("Volume by Region & Timepoint"),
          plotOutput("volume_bar", height = "380px")
        ),
        div(class = "dash-card",
          h4("Volume Distribution"),
          plotOutput("volume_box", height = "380px")
        )
      ),
      div(class = "dash-card",
        h4("Region Detail Table"),
        DTOutput("region_dt")
      )
    ),

    # ---- Tab 4: Tracking ----
    nav_panel(
      title = "Tracking",
      icon  = icon("route"),
      br(),
      div(style = "color:#a0888e; font-size:13px; margin-bottom:12px; padding: 0 4px;",
        "Regions are matched across consecutive timepoints using spatial overlap (IoU), centroid proximity, and volume similarity. ",
        tags$b("CONTINUATION"), " = same region persists. ",
        tags$b("APPEARANCE"), " = new region found. ",
        tags$b("DISAPPEARANCE"), " = region gone. ",
        tags$b("SPLIT"), " = one region broke into multiple. ",
        tags$b("MERGE"), " = multiple regions fused into one. ",
        "These events reveal tumor behavior that total volume alone cannot capture."
      ),
      layout_columns(
        col_widths = c(5, 7),
        div(class = "dash-card",
          h4("Event Summary"),
          plotOutput("event_bar", height = "340px")
        ),
        div(class = "dash-card",
          h4("Event Detail"),
          DTOutput("event_dt")
        )
      )
    ),

    # ---- Tab 5: Progression ----
    nav_panel(
      title = "Progression",
      icon  = icon("chart-line"),
      br(),
      div(style = "color:#a0888e; font-size:13px; margin-bottom:12px; padding: 0 4px;",
        "Volume trajectories show how each tumor region changes over time. ",
        "The system classifies progression using RANO criteria: ",
        tags$b("CR"), " = all tumor gone, ",
        tags$b("PR"), " = shrinking, ",
        tags$b("SD"), " = stable, ",
        tags$b("PD"), " = growing or new lesions. ",
        "Region-wise analysis catches hidden changes — e.g. edema shrinks while enhancing grows, making total volume look stable when the tumor is actually progressing."
      ),
      div(class = "dash-card",
        plotOutput("trajectory_plot", height = "380px")
      ),
      layout_columns(
        col_widths = c(6, 6),
        div(class = "dash-card", htmlOutput("progression_card")),
        div(class = "dash-card",
          h4("Trajectory Detail"),
          DTOutput("traj_dt")
        )
      )
    ),

    # ---- Tab 6: Report ----
    nav_panel(
      title = "Report",
      icon  = icon("file-medical"),
      br(),
      div(class = "dash-card",
        h4("Full Patient Report"),
        tags$p(style = "color:#a0888e; font-size:13px; margin-bottom:12px;",
          "Complete text report summarizing all findings — classification, region trajectories, events, and evidence. Copy this for documentation."
        ),
        verbatimTextOutput("full_report")
      )
    ),

    # ---- Tab 7: Pipeline Gallery ----
    nav_panel(
      title = "Pipeline Results",
      icon  = icon("images"),
      br(),
      layout_columns(
        col_widths = c(8, 4),
        div(class = "dash-card",
          h4("Generated Figures"),
          uiOutput("fig_selector"),
          imageOutput("selected_fig", height = "auto")
        ),
        div(class = "dash-card",
          h4("Result Tables"),
          DTOutput("pipeline_tables_dt"),
          br(),
          uiOutput("table_download_ui")
        )
      )
    )
  ),

  # welcome overlay
  conditionalPanel(
    condition = "!output.has_results",
    div(class = "welcome-screen",
      tags$span(style = "font-size:56px; opacity:0.3;", "\U0001F9E0"),
      h3("Select a patient and click Analyze"),
      p("The pipeline will load MRI, run 3D U-Net segmentation,"),
      p("track regions longitudinally, and classify progression.")
    )
  )
)

# ── Server ────────────────────────────────────────────────────

server <- function(input, output, session) {

  rv <- reactiveValues(
    results = NULL, patient_data = NULL, preprocessed = NULL,
    region_data = NULL, events = NULL, trajectories = NULL,
    progression = NULL, status = ""
  )

  output$has_results <- reactive(!is.null(rv$results))
  outputOptions(output, "has_results", suspendWhenHidden = FALSE)

  # ── Run analysis ──

  observeEvent(input$run_btn, {
    if (input$input_mode == "custom") {
      req(input$custom_path)
      patient_path <- normalizePath(input$custom_path, winslash = "/", mustWork = FALSE)
      pid <- basename(patient_path)
    } else {
      req(input$patient_id)
      pid <- input$patient_id
      patient_path <- file.path(IMAGING_ROOT, pid)
    }
    if (!dir.exists(patient_path)) { rv$status <- paste("Folder not found:", patient_path); return() }

    rv$results <- NULL
    withProgress(message = paste("Analyzing", pid), value = 0, {

      incProgress(0.10, detail = "Loading MRI...")
      rv$status <- "Loading MRI scans..."
      patient_data <- tryCatch(load_full_patient(patient_path),
                               error = function(e) { rv$status <- paste("Load error:", e$message); NULL })
      if (is.null(patient_data)) return()
      rv$patient_data <- patient_data

      incProgress(0.20, detail = "Preprocessing...")
      rv$status <- "Preprocessing (normalize, resize to 48³)..."
      preprocessed <- tryCatch(preprocess_full_patient(patient_data, target_size = TARGET_SIZE),
                               error = function(e) { rv$status <- paste("Error:", e$message); NULL })
      if (is.null(preprocessed)) return()
      rv$preprocessed <- preprocessed

      incProgress(0.20, detail = "3D U-Net segmentation...")
      rv$status <- "Running 3D U-Net segmentation..."
      model <- tryCatch(load_trained_unet(MODEL_PATH, "cpu"), error = function(e) NULL)
      if (is.null(model)) { rv$status <- "Model load failed"; return() }
      segmented <- segment_patient(model, preprocessed, "cpu")
      rm(model); gc()

      incProgress(0.15, detail = "Extracting regions...")
      rv$status <- "Extracting tumor regions..."
      region_data <- extract_all_patient_regions(segmented, pid, min_vol = 50)
      rv$region_data <- region_data

      incProgress(0.10, detail = "Tracking regions...")
      rv$status <- "Tracking regions across timepoints..."
      events <- NULL; trajectories <- NULL; progression <- NULL

      if (length(region_data$regions_by_timepoint) >= 2) {
        tracking <- track_patient_regions(region_data$regions_by_timepoint)
        events <- create_event_table(tracking$all_events, pid)

        tp_names <- names(region_data$regions_by_timepoint)
        time_weeks <- sapply(patient_data$timepoints[tp_names], `[[`, "week_number")
        trajectories <- list()
        for (r in region_data$regions_by_timepoint[[1]]) {
          vols <- sapply(tp_names, function(tp) {
            matched <- Filter(function(x) x$region_type == r$region_type,
                              region_data$regions_by_timepoint[[tp]])
            if (length(matched) > 0) matched[[1]]$volume_cm3 else 0
          })
          trajectories[[r$region_id]] <- analyze_region_trajectory(vols, time_weeks)
        }
        if (is.null(events)) events <- tibble(event_type = character(0))
        progression <- classify_patient_progression(trajectories, events)
      }
      rv$events <- events; rv$trajectories <- trajectories; rv$progression <- progression

      incProgress(0.15, detail = "Generating visualizations...")
      rv$results <- list(pid = pid, segmented = segmented,
                         region_table = region_data$table, preprocessed = preprocessed)
      rv$status <- paste("Analysis complete for", pid)
    })
  })

  # ── Sidebar outputs ──

  output$status_text <- renderUI({
    div(class = "status-msg", rv$status)
  })

  output$patient_banner <- renderUI({
    req(rv$results, rv$patient_data)
    pd <- rv$patient_data
    tp_names <- names(pd$timepoints)
    n_tp <- length(tp_names)
    weeks <- sapply(pd$timepoints, `[[`, "week_number")
    span_weeks <- if (length(weeks) >= 2) paste0(min(weeks), " - ", max(weeks), " weeks") else paste0(weeks[1], " weeks")

    n_regions <- if (!is.null(rv$region_data)) nrow(rv$region_data$table) else 0
    n_events <- if (!is.null(rv$events)) nrow(rv$events) else 0

    modalities <- c()
    first_tp <- pd$timepoints[[1]]
    if (!is.null(first_tp$mri$available)) modalities <- toupper(first_tp$mri$available)
    mod_str <- if (length(modalities) > 0) paste(modalities, collapse = ", ") else "T1, T1C, T2, FLAIR"

    has_seg <- !is.null(first_tp$segmentation$mask)
    seg_str <- if (has_seg) "Ground truth available" else "U-Net predicted"

    prog <- rv$progression
    prog_str <- if (!is.null(prog)) paste0(prog$classification, " (", round(prog$confidence*100), "% confidence)") else "Not classified"

    HTML(paste0(
      "<div class='patient-banner'>",
      "<h3>\U0001F9E0 ", pd$patient_id, " — Longitudinal Brain MRI Analysis</h3>",
      "<div class='pb-grid'>",
      "<div class='pb-item'><div class='pb-label'>Timepoints</div><div class='pb-value'>", n_tp, " scans (", span_weeks, ")</div></div>",
      "<div class='pb-item'><div class='pb-label'>MRI Modalities</div><div class='pb-value'>", mod_str, "</div></div>",
      "<div class='pb-item'><div class='pb-label'>Segmentation</div><div class='pb-value'>", seg_str, "</div></div>",
      "<div class='pb-item'><div class='pb-label'>Progression</div><div class='pb-value'>", prog_str, "</div></div>",
      "</div>",
      "<div style='margin-top:12px; font-size:12px; color:#a0888e;'>",
      "Scans: ", paste(tp_names, collapse = " &rarr; "),
      " &nbsp;|&nbsp; Regions: ", n_regions,
      " &nbsp;|&nbsp; Events: ", n_events,
      "</div>",
      "</div>"
    ))
  })

  output$patient_clinical_summary <- renderUI({
    req(rv$results, rv$region_data, rv$patient_data)
    pd <- rv$patient_data
    rt <- rv$region_data$table
    prog <- rv$progression
    events <- rv$events
    trajs <- rv$trajectories

    pid <- pd$patient_id
    tp_names <- names(pd$timepoints)
    n_tp <- length(tp_names)
    weeks <- sapply(pd$timepoints, `[[`, "week_number")

    n_total_regions <- nrow(rt)
    region_types <- rt %>% distinct(region_type) %>% pull(region_type)
    n_types <- length(region_types)

    regions_per_tp <- rt %>% group_by(timepoint) %>%
      summarise(n = n(), total_vol = sum(volume_cm3, na.rm = TRUE), .groups = "drop")

    first_tp <- regions_per_tp %>% slice(1)
    last_tp <- regions_per_tp %>% slice(n())

    largest_region <- rt %>% arrange(desc(volume_cm3)) %>% slice(1)

    location_desc <- if (nrow(largest_region) > 0) {
      cx <- round(largest_region$centroid_x, 0)
      cy <- round(largest_region$centroid_y, 0)
      cz <- round(largest_region$centroid_z, 0)
      paste0("Centroid at voxel (", cx, ", ", cy, ", ", cz, ") in 48x48x48 space")
    } else "Unknown"

    type_summary <- rt %>% group_by(region_type) %>%
      summarise(count = n(), avg_vol = round(mean(volume_cm3, na.rm=TRUE), 2),
                max_vol = round(max(volume_cm3, na.rm=TRUE), 2), .groups = "drop")

    type_rows <- paste0(apply(type_summary, 1, function(r) {
      col <- switch(r["region_type"], necrosis="#ef4444", edema="#f59e0b", enhancing="#22c55e", "#60a5fa")
      paste0("<tr><td><span style='color:", col, "; font-weight:700;'>",
             toupper(r["region_type"]), "</span></td><td>", r["count"],
             "</td><td>", r["avg_vol"], " cm&sup3;</td><td>", r["max_vol"], " cm&sup3;</td></tr>")
    }), collapse = "")

    n_events <- if (!is.null(events)) nrow(events) else 0
    n_appear <- if (!is.null(events)) sum(events$event_type == "APPEARANCE") else 0
    n_disappear <- if (!is.null(events)) sum(events$event_type == "DISAPPEARANCE") else 0
    n_cont <- if (!is.null(events)) sum(events$event_type == "CONTINUATION") else 0
    n_split <- if (!is.null(events)) sum(events$event_type == "SPLIT") else 0
    n_merge <- if (!is.null(events)) sum(events$event_type == "MERGE") else 0

    vol_change_pct <- if (first_tp$total_vol > 0) {
      round((last_tp$total_vol - first_tp$total_vol) / first_tp$total_vol * 100, 1)
    } else NA
    vol_arrow <- if (!is.na(vol_change_pct) && vol_change_pct > 5) "\U0001F53A"
                 else if (!is.na(vol_change_pct) && vol_change_pct < -5) "\U0001F53B"
                 else "\U000027A1"

    prog_cls <- if (!is.null(prog)) prog$classification else "N/A"
    prog_conf <- if (!is.null(prog)) round(prog$confidence * 100) else 0
    prog_color <- switch(prog_cls, CR="#22c55e", PR="#60a5fa", SD="#f59e0b", PD="#ef4444", "#a0888e")
    prog_label <- switch(prog_cls,
      CR="Complete Response — All tumor regions have disappeared. Treatment appears highly effective.",
      PR="Partial Response — Tumor regions are shrinking or disappearing. Positive treatment response.",
      SD="Stable Disease — No significant change in tumor size. Disease is controlled but not regressing.",
      PD="Progressive Disease — Tumor regions are growing or new lesions appeared. May require treatment adjustment.",
      "Not classified")

    traj_rows <- ""
    if (!is.null(trajs)) {
      traj_rows <- paste0(sapply(names(trajs), function(rid) {
        t <- trajs[[rid]]
        pct <- if (!is.na(t$percent_change)) sprintf("%+.0f%%", t$percent_change * 100) else "N/A"
        trend_col <- switch(t$trend, GROWTH="#ef4444", SHRINKAGE="#22c55e", STABLE="#f59e0b",
                            DISAPPEARED="#a855f7", APPEARED="#ef4444", "#a0888e")
        paste0("<tr><td>", rid, "</td><td><span style='color:", trend_col, "; font-weight:700;'>",
               t$trend, "</span></td><td>", round(t$first_volume, 1), " cm&sup3;</td><td>",
               round(t$last_volume, 1), " cm&sup3;</td><td>", pct, "</td></tr>")
      }), collapse = "")
    }

    HTML(paste0(
      # Overview card
      "<div class='dash-card'>",
      "<h4 style='font-size:18px; color:#fff;'>\U0001F9E0 Clinical Overview — ", pid, "</h4>",
      "<div style='display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; margin-top:14px;'>",

      "<div class='dash-card' style='padding:14px; margin:0; border-left:3px solid #8b1a2b;'>",
      "<div style='color:#a0888e; font-size:11px; text-transform:uppercase;'>Scan Timeline</div>",
      "<div style='color:#fff; font-size:15px; font-weight:700; margin-top:4px;'>", n_tp, " MRI scans</div>",
      "<div style='color:#c0a8ae; font-size:12px;'>Week ", min(weeks), " to Week ", max(weeks), " (", max(weeks)-min(weeks), " week span)</div>",
      "<div style='color:#a0888e; font-size:12px; margin-top:4px;'>", paste(tp_names, collapse = " &rarr; "), "</div>",
      "</div>",

      "<div class='dash-card' style='padding:14px; margin:0; border-left:3px solid #8b1a2b;'>",
      "<div style='color:#a0888e; font-size:11px; text-transform:uppercase;'>Total Tumor Volume</div>",
      "<div style='color:#fff; font-size:15px; font-weight:700; margin-top:4px;'>",
      round(first_tp$total_vol, 1), " &rarr; ", round(last_tp$total_vol, 1), " cm&sup3; ", vol_arrow, "</div>",
      "<div style='color:#c0a8ae; font-size:12px;'>Change: ",
      if (!is.na(vol_change_pct)) paste0(sprintf("%+.1f", vol_change_pct), "%") else "N/A", "</div>",
      "</div>",

      "<div class='dash-card' style='padding:14px; margin:0; border-left:3px solid ", prog_color, ";'>",
      "<div style='color:#a0888e; font-size:11px; text-transform:uppercase;'>RANO Classification</div>",
      "<div style='color:", prog_color, "; font-size:22px; font-weight:800; margin-top:4px;'>", prog_cls, "</div>",
      "<div style='color:#c0a8ae; font-size:12px;'>Confidence: ", prog_conf, "%</div>",
      "</div>",

      "</div></div>",

      # Progression interpretation
      "<div class='dash-card'>",
      "<h4>\U0001F4CB Diagnosis Interpretation</h4>",
      "<p style='color:#e0d0d5; font-size:14px; line-height:1.7;'>",
      "<span style='color:", prog_color, "; font-weight:700; font-size:16px;'>", prog_cls, "</span> &mdash; ", prog_label,
      "</p></div>",

      # Tumor regions breakdown
      "<div class='dash-card'>",
      "<h4>\U0001F3AF Tumor Regions Found</h4>",
      "<p style='color:#a0888e; font-size:13px;'>",
      n_total_regions, " total regions detected across ", n_tp, " timepoints. ",
      "Region types: ", paste(toupper(region_types), collapse = ", "), ". ",
      "Largest region: <b>", toupper(largest_region$region_type), "</b> at <b>",
      round(largest_region$volume_cm3, 1), " cm&sup3;</b> (", largest_region$timepoint, "). ",
      location_desc, ".</p>",
      "<table class='table' style='color:#e0d0d5; font-size:13px; margin-top:10px;'>",
      "<thead><tr><th style='color:#a0888e;'>Region Type</th><th style='color:#a0888e;'>Count</th>",
      "<th style='color:#a0888e;'>Avg Volume</th><th style='color:#a0888e;'>Max Volume</th></tr></thead>",
      "<tbody>", type_rows, "</tbody></table>",
      "</div>",

      # Region trajectories
      if (!is.null(trajs) && length(trajs) > 0) paste0(
        "<div class='dash-card'>",
        "<h4>\U0001F4C8 Region Trajectories</h4>",
        "<p style='color:#a0888e; font-size:13px;'>How each tracked region changed from first to last scan.</p>",
        "<table class='table' style='color:#e0d0d5; font-size:13px;'>",
        "<thead><tr><th style='color:#a0888e;'>Region</th><th style='color:#a0888e;'>Trend</th>",
        "<th style='color:#a0888e;'>First Vol</th><th style='color:#a0888e;'>Last Vol</th>",
        "<th style='color:#a0888e;'>Change</th></tr></thead>",
        "<tbody>", traj_rows, "</tbody></table></div>"
      ) else "",

      # Events summary
      "<div class='dash-card'>",
      "<h4>\U0001F504 Longitudinal Events</h4>",
      "<p style='color:#a0888e; font-size:13px;'>",
      n_events, " total events detected: ",
      n_cont, " continuations, ",
      n_appear, " new appearances, ",
      n_disappear, " disappearances, ",
      n_split, " splits, ",
      n_merge, " merges.",
      if (n_disappear > 0) paste0(" <span style='color:#22c55e;'>\U00002705 ", n_disappear, " region(s) disappeared — positive sign.</span>") else "",
      if (n_appear > 0) paste0(" <span style='color:#ef4444;'>\U000026A0 ", n_appear, " new region(s) appeared — concerning.</span>") else "",
      "</p></div>"
    ))
  })

  output$patient_info <- renderUI({
    pd <- rv$patient_data
    if (is.null(pd)) return(div(style = "color:#475569; font-size:13px;", "No patient loaded"))
    tp_names <- names(pd$timepoints)
    rows <- function(lbl, val) {
      paste0("<div class='info-row'><span class='info-label'>", lbl,
             "</span><span class='info-value'>", val, "</span></div>")
    }
    HTML(paste0(
      rows("Patient", pd$patient_id),
      rows("Timepoints", length(tp_names)),
      rows("Scans", paste(tp_names, collapse = ", "))
    ))
  })

  # ── Stat cards ──

  output$stat_progression <- renderUI({
    prog <- rv$progression
    if (is.null(prog)) return(HTML(
      "<div class='stat-icon'>\U0001F4CA</div><div class='stat-value' style='color:#475569;'>--</div><div class='stat-label'>Classification</div>"))
    cls <- prog$classification
    label <- switch(cls, CR="Complete Response", PR="Partial Response",
                    SD="Stable Disease", PD="Progressive Disease", cls)
    HTML(paste0(
      "<div class='stat-icon'>\U0001F3AF</div>",
      "<div class='stat-value clr-", tolower(cls), "'>", cls, "</div>",
      "<div class='stat-label'>", label, "</div>"))
  })

  output$stat_regions <- renderUI({
    n <- if (!is.null(rv$region_data)) nrow(rv$region_data$table) else "--"
    HTML(paste0("<div class='stat-icon'>\U0001F9EC</div><div class='stat-value'>", n,
                "</div><div class='stat-label'>Regions Found</div>"))
  })

  output$stat_events <- renderUI({
    n <- if (!is.null(rv$events) && nrow(rv$events) > 0) nrow(rv$events) else "--"
    HTML(paste0("<div class='stat-icon'>\U0001F50D</div><div class='stat-value'>", n,
                "</div><div class='stat-label'>Events Detected</div>"))
  })

  output$stat_confidence <- renderUI({
    prog <- rv$progression
    v <- if (!is.null(prog)) paste0(round(prog$confidence * 100), "%") else "--"
    HTML(paste0("<div class='stat-icon'>✅</div><div class='stat-value'>", v,
                "</div><div class='stat-label'>Confidence</div>"))
  })

  # ── Tab 1: All MRI Modalities ──

  output$mri_all_modalities <- renderPlot({
    req(rv$results)
    pp <- rv$results$preprocessed
    tp_names <- names(pp)
    mod_names <- c("T1", "T1c", "T2", "FLAIR")
    plots <- list()

    for (tp in tp_names[seq_len(min(length(tp_names), 4))]) {
      stk <- pp[[tp]]$stacked
      if (is.null(stk)) next
      n_ch <- dim(stk)[4]
      for (ch in seq_len(n_ch)) {
        vol <- stk[, , , ch]
        z <- dim(vol)[3] %/% 2
        sl <- vol[, , z]
        df <- expand.grid(x = seq_len(nrow(sl)), y = seq_len(ncol(sl)))
        df$v <- as.vector(sl)
        mod_label <- if (ch <= length(mod_names)) mod_names[ch] else paste0("Ch", ch)
        p <- ggplot(df, aes(x, y, fill = v)) +
          geom_raster() +
          scale_fill_viridis_c(option = "inferno", guide = "none") +
          coord_fixed() +
          labs(title = paste(tp, "—", mod_label)) +
          theme_void() +
          theme(plot.title = element_text(hjust = 0.5, size = 10, color = "#bdbdbd", face = "bold"),
                plot.background = element_rect(fill = "#1a1a2e", color = NA),
                plot.margin = margin(2, 2, 2, 2))
        plots <- c(plots, list(p))
      }
    }
    if (length(plots) == 0) return(NULL)
    ncols <- min(dim(pp[[1]]$stacked)[4], 4)
    wrap_plots(plots, ncol = ncols) +
      plot_annotation(
        title = paste("Multi-Modal MRI —", rv$results$pid),
        subtitle = "Axial mid-slice | Rows = timepoints, Columns = modalities",
        theme = theme(plot.title = element_text(face = "bold", size = 16, color = "#fff"),
                      plot.subtitle = element_text(color = "#9e9e9e", size = 11),
                      plot.background = element_rect(fill = "#1a1a2e", color = NA)))
  }, bg = "#1c1014")

  # ── Tab 2: Segmentation mask ──

  output$seg_plot <- renderPlot({
    req(rv$results)
    seg_data <- rv$results$segmented
    tp_names <- names(seg_data)
    plots <- list()
    for (tp in tp_names[seq_len(min(length(tp_names), 4))]) {
      seg <- seg_data[[tp]]$segmentation
      if (is.null(seg)) next
      z <- dim(seg)[3] %/% 2; sl <- seg[, , z]
      df <- expand.grid(x = seq_len(nrow(sl)), y = seq_len(ncol(sl)))
      df$label <- factor(as.vector(sl))
      p <- ggplot(df, aes(x, y, fill = label)) +
        geom_raster() +
        scale_fill_manual(values = SEG_PAL, labels = SEG_LABS, drop = FALSE, guide = "none") +
        coord_fixed() + labs(title = tp) +
        theme_void() +
        theme(plot.title = element_text(hjust = .5, size = 11, color = "#bdbdbd", face = "bold"),
              plot.background = element_rect(fill = "#1a1a2e", color = NA))
      plots <- c(plots, list(p))
    }
    if (length(plots) == 0) return(NULL)
    wrap_plots(plots, ncol = min(4, length(plots))) +
      plot_annotation(title = "Segmentation Mask",
        theme = theme(plot.title = element_text(face = "bold", size = 14, color = "#fff"),
                      plot.background = element_rect(fill = "#1a1a2e", color = NA)))
  }, bg = "#1c1014")

  # ── Tab 2: Overlay ──

  output$overlay_plot <- renderPlot({
    req(rv$results)
    seg_data <- rv$results$segmented
    pp <- rv$results$preprocessed
    tp_names <- names(seg_data)
    plots <- list()

    seg_rgb <- list("1" = c(239,68,68), "2" = c(245,158,11), "3" = c(34,197,94), "4" = c(96,165,250))

    for (tp in tp_names[seq_len(min(length(tp_names), 4))]) {
      seg <- seg_data[[tp]]$segmentation
      stk <- pp[[tp]]$stacked
      if (is.null(seg) || is.null(stk)) next

      flair_ch <- min(dim(stk)[4], 4)
      flair <- stk[, , , flair_ch]
      z <- dim(flair)[3] %/% 2
      fl_sl <- flair[, , z]; sg_sl <- seg[, , z]
      nr <- nrow(fl_sl); nc <- ncol(fl_sl)

      fl_norm <- (fl_sl - min(fl_sl)) / max(max(fl_sl) - min(fl_sl), 1e-8)
      r_ch <- fl_norm; g_ch <- fl_norm; b_ch <- fl_norm
      alpha <- 0.55
      for (lbl in names(seg_rgb)) {
        mask <- sg_sl == as.integer(lbl)
        if (!any(mask)) next
        rgb_v <- seg_rgb[[lbl]] / 255
        r_ch[mask] <- r_ch[mask] * (1 - alpha) + rgb_v[1] * alpha
        g_ch[mask] <- g_ch[mask] * (1 - alpha) + rgb_v[2] * alpha
        b_ch[mask] <- b_ch[mask] * (1 - alpha) + rgb_v[3] * alpha
      }
      hex_vals <- rgb(as.vector(r_ch), as.vector(g_ch), as.vector(b_ch))

      df <- expand.grid(x = seq_len(nr), y = seq_len(nc))
      df$color <- hex_vals

      p <- ggplot(df, aes(x, y)) +
        geom_raster(fill = df$color) +
        coord_fixed() + labs(title = tp) +
        theme_void() +
        theme(plot.title = element_text(hjust = .5, size = 11, color = "#bdbdbd", face = "bold"),
              plot.background = element_rect(fill = "#1a1a2e", color = NA))
      plots <- c(plots, list(p))
    }
    if (length(plots) == 0) return(NULL)
    wrap_plots(plots, ncol = min(4, length(plots))) +
      plot_annotation(title = "FLAIR + Segmentation Overlay",
        theme = theme(plot.title = element_text(face = "bold", size = 14, color = "#fff"),
                      plot.background = element_rect(fill = "#1a1a2e", color = NA)))
  }, bg = "#1c1014")

  # ── Tab 3: Volume bar ──

  output$volume_bar <- renderPlot({
    req(rv$results)
    rt <- rv$results$region_table
    if (nrow(rt) == 0) return(NULL)
    ggplot(rt, aes(timepoint, volume_cm3, fill = region_type)) +
      geom_col(position = "dodge", alpha = .88, width = .55) +
      geom_text(aes(label = round(volume_cm3, 1)),
                position = position_dodge(.55), vjust = -.6, size = 3.3, color = "#e0e0e0") +
      scale_fill_manual(values = REGION_PAL, name = "Region") +
      labs(title = "Region Volumes", x = NULL, y = "Volume (cm³)") +
      dark_theme()
  }, bg = "#1c1014")

  output$volume_box <- renderPlot({
    req(rv$results)
    rt <- rv$results$region_table
    if (nrow(rt) == 0) return(NULL)
    ggplot(rt, aes(region_type, volume_cm3, fill = region_type)) +
      geom_boxplot(alpha = .7, outlier.color = "#e0e0e0", width = .5) +
      geom_jitter(width = .15, alpha = .5, size = 2, color = "#e0e0e0") +
      scale_fill_manual(values = REGION_PAL, guide = "none") +
      labs(title = "Distribution", x = NULL, y = "Volume (cm³)") +
      dark_theme()
  }, bg = "#1c1014")

  output$region_dt <- renderDT({
    req(rv$results)
    rt <- rv$results$region_table
    if (nrow(rt) == 0) return(NULL)
    rt %>%
      select(timepoint, region_id, region_type, volume_cm3, n_voxels,
             centroid_x, centroid_y, centroid_z) %>%
      mutate(across(c(volume_cm3, centroid_x, centroid_y, centroid_z), ~round(., 2))) %>%
      datatable(options = list(pageLength = 10, dom = "ftp",
                               initComplete = JS("function(s,d){$(this.api().table().header()).css({'color':'#94a3b8'});}")),
                class = "compact stripe", rownames = FALSE)
  })

  # ── Tab 4: Events ──

  output$event_bar <- renderPlot({
    req(rv$events)
    ev <- rv$events
    if (is.null(ev) || nrow(ev) == 0) return(NULL)
    ev %>% count(event_type) %>%
      mutate(event_type = fct_reorder(event_type, n, .desc = TRUE)) %>%
      ggplot(aes(event_type, n, fill = event_type)) +
      geom_col(alpha = .88, width = .55) +
      geom_text(aes(label = n), vjust = -.6, fontface = "bold", size = 5, color = "#e0e0e0") +
      scale_fill_manual(values = EVENT_PAL, guide = "none") +
      labs(title = "Region Events", x = NULL, y = "Count") +
      dark_theme()
  }, bg = "#1c1014")

  output$event_dt <- renderDT({
    req(rv$events)
    ev <- rv$events
    if (is.null(ev) || nrow(ev) == 0) return(NULL)
    ev %>% select(event_type, any_of(c("from_tp","to_tp","from_region","to_region","region","score"))) %>%
      datatable(options = list(pageLength = 10, dom = "ftp",
                               initComplete = JS("function(s,d){$(this.api().table().header()).css({'color':'#94a3b8'});}")),
                class = "compact stripe", rownames = FALSE)
  })

  # ── Tab 5: Progression ──

  output$trajectory_plot <- renderPlot({
    req(rv$results, rv$trajectories)
    rt <- rv$results$region_table
    if (nrow(rt) == 0) return(NULL)

    total <- rt %>% group_by(timepoint) %>%
      summarise(vol = sum(volume_cm3, na.rm = TRUE), .groups = "drop")

    p1 <- ggplot(rt, aes(timepoint, volume_cm3, color = region_type, group = region_type)) +
      geom_line(linewidth = 1.4) + geom_point(size = 3.5) +
      scale_color_manual(values = REGION_PAL, name = "Region") +
      labs(title = "Region Trajectories", x = NULL, y = "Volume (cm³)") +
      dark_theme()

    p2 <- ggplot(total, aes(timepoint, vol, group = 1)) +
      geom_area(alpha = .15, fill = "#60a5fa") +
      geom_line(linewidth = 1.4, color = "#60a5fa") +
      geom_point(size = 3.5, color = "#60a5fa") +
      geom_text(aes(label = round(vol, 1)), vjust = -1.2, color = "#e0e0e0", size = 3.8) +
      labs(title = "Total Volume", x = NULL, y = "cm³") +
      dark_theme()

    p1 + p2 + plot_layout(widths = c(1.3, 1))
  }, bg = "#1c1014")

  output$progression_card <- renderUI({
    prog <- rv$progression
    if (is.null(prog)) return(NULL)
    cls <- prog$classification; conf <- round(prog$confidence * 100)
    clr <- RANO_PAL[cls]
    label <- switch(cls,
      CR="Complete Response — All tumor regions disappeared",
      PR="Partial Response — Tumor regions are shrinking",
      SD="Stable Disease — No significant change",
      PD="Progressive Disease — Tumor regions are growing", cls)
    ev <- prog$evidence
    items <- c()
    if (ev$n_growing > 0)     items <- c(items, paste0(ev$n_growing, " growing region(s)"))
    if (ev$n_shrinking > 0)   items <- c(items, paste0(ev$n_shrinking, " shrinking region(s)"))
    if (ev$n_stable > 0)      items <- c(items, paste0(ev$n_stable, " stable region(s)"))
    if (ev$n_appeared > 0)    items <- c(items, paste0(ev$n_appeared, " new region(s)"))
    if (ev$n_disappeared > 0) items <- c(items, paste0(ev$n_disappeared, " disappeared"))
    if (ev$has_splits)        items <- c(items, "Split event detected")

    HTML(paste0(
      "<h4 style='color:", clr, "; margin-top:0; font-size:22px;'>", cls, "</h4>",
      "<p style='color:#bdbdbd; margin-bottom:12px;'>", label, "</p>",
      "<div style='display:flex; gap:24px; margin-bottom:14px;'>",
      "<div><span style='font-size:28px; font-weight:800; color:", clr, ";'>", conf, "%</span>",
      "<br><span style='font-size:11px; color:#94a3b8;'>CONFIDENCE</span></div></div>",
      "<div style='border-top:1px solid #334155; padding-top:10px;'>",
      "<span style='font-size:12px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.8px;'>Evidence</span>",
      "<ul style='margin:6px 0; padding-left:18px; color:#bdbdbd; font-size:13px;'>",
      paste0("<li>", items, "</li>", collapse = ""),
      "</ul></div>"
    ))
  })

  output$traj_dt <- renderDT({
    req(rv$trajectories)
    df <- map_dfr(names(rv$trajectories), function(rid) {
      t <- rv$trajectories[[rid]]
      tibble(Region = rid, Trend = t$trend,
             `First (cm3)` = round(t$first_volume, 2),
             `Last (cm3)` = round(t$last_volume, 2),
             Change = if (!is.na(t$percent_change)) sprintf("%+.0f%%", t$percent_change * 100) else "N/A")
    })
    datatable(df, options = list(dom = "t", pageLength = 20,
              initComplete = JS("function(s,d){$(this.api().table().header()).css({'color':'#94a3b8'});}")),
              class = "compact stripe", rownames = FALSE)
  })

  # ── Tab 6: Report ──

  output$full_report <- renderText({
    req(rv$results, rv$progression)
    pid <- rv$results$pid; rt <- rv$results$region_table; prog <- rv$progression
    lines <- c(
      strrep("═", 60),
      paste("PATIENT REPORT:", pid),
      strrep("═", 60),
      paste("Date:", Sys.Date()),
      "Model: 3D U-Net (5.6M params, JIT) + Rule-based RANO", "",
      paste("CLASSIFICATION:", prog$classification),
      paste("CONFIDENCE:    ", paste0(round(prog$confidence * 100), "%")), "",
      paste("Regions found: ", nrow(rt)),
      paste("Timepoints:    ", paste(unique(rt$timepoint), collapse = ", ")))

    if (!is.null(rv$trajectories)) {
      lines <- c(lines, "", "REGION TRAJECTORIES:")
      for (rid in names(rv$trajectories)) {
        t <- rv$trajectories[[rid]]
        pct <- if (!is.na(t$percent_change)) sprintf("%+.0f%%", t$percent_change * 100) else "N/A"
        lines <- c(lines, sprintf("  %-8s %-12s %6.1f -> %6.1f cm3  %s",
                                  rid, t$trend, t$first_volume, t$last_volume, pct))
      }
    }
    if (!is.null(rv$events) && nrow(rv$events) > 0) {
      lines <- c(lines, "", "EVENTS:")
      for (i in seq_len(nrow(rv$events))) {
        e <- rv$events[i, ]
        lines <- c(lines, sprintf("  [%s] %s -> %s",
          e$event_type,
          if ("from_tp" %in% names(e)) e$from_tp else "",
          if ("to_tp" %in% names(e)) e$to_tp else ""))
      }
    }
    ev <- prog$evidence
    lines <- c(lines, "",
      paste("Growing:", ev$n_growing, "| Shrinking:", ev$n_shrinking,
            "| Stable:", ev$n_stable, "| Appeared:", ev$n_appeared,
            "| Disappeared:", ev$n_disappeared),
      "", strrep("═", 60))
    paste(lines, collapse = "\n")
  })

  # ── Tab 7: Pipeline Gallery ──

  output$fig_selector <- renderUI({
    if (length(existing_figs) == 0) return(p("No pipeline figures found.", style = "color:#64748b;"))
    selectInput("chosen_fig", NULL, choices = existing_figs, selected = existing_figs[1], width = "100%")
  })

  output$selected_fig <- renderImage({
    req(input$chosen_fig)
    path <- file.path(FIGURES_PATH, input$chosen_fig)
    list(src = path, contentType = "image/png", width = "100%", alt = input$chosen_fig)
  }, deleteFile = FALSE)

  output$pipeline_tables_dt <- renderDT({
    if (length(existing_tables) == 0) return(NULL)
    tibble(File = existing_tables,
           Size = file.size(file.path(TABLES_PATH, existing_tables)) %>%
             {ifelse(. > 1e6, paste0(round(./1e6,1),"MB"), paste0(round(./1e3,1),"KB"))}) %>%
      datatable(options = list(dom = "t", pageLength = 20,
                initComplete = JS("function(s,d){$(this.api().table().header()).css({'color':'#94a3b8'});}")),
                selection = "single", class = "compact stripe", rownames = FALSE)
  })

  output$table_download_ui <- renderUI({
    sel <- input$pipeline_tables_dt_rows_selected
    if (is.null(sel) || length(sel) == 0) return(NULL)
    fname <- existing_tables[sel]
    downloadButton("dl_table", paste("Download", fname), class = "run-btn",
                   style = "font-size:13px; padding:10px 16px;")
  })

  output$dl_table <- downloadHandler(
    filename = function() {
      sel <- input$pipeline_tables_dt_rows_selected
      if (is.null(sel)) "data.csv" else existing_tables[sel]
    },
    content = function(file) {
      sel <- input$pipeline_tables_dt_rows_selected
      if (!is.null(sel)) file.copy(file.path(TABLES_PATH, existing_tables[sel]), file)
    }
  )
}

shinyApp(ui = ui, server = server)
