## ══════════════════════════════════════════════════════════════
## Tumor Progression Analysis — Neon Blue Dashboard
## Run:
##   cd "C:/Users/mail2/OneDrive/Desktop/R project 1"
##   Rscript -e "shiny::runApp('app.R', launch.browser=TRUE)"
## ══════════════════════════════════════════════════════════════

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
FIGURES_PATH <- file.path(PROJECT_ROOT, "results", "figures")
TABLES_PATH  <- file.path(PROJECT_ROOT, "results", "tables")
TARGET_SIZE  <- c(48L, 48L, 48L)

patient_list <- sort(list.dirs(IMAGING_ROOT, recursive = FALSE, full.names = FALSE))
patient_list <- patient_list[grepl("^Patient-", patient_list)]

TEST_DATA_ROOT <- normalizePath(file.path(Sys.getenv("USERPROFILE"),
  "OneDrive/Desktop/test_brain_mri"), winslash = "/", mustWork = FALSE)
test_list <- if (dir.exists(TEST_DATA_ROOT)) {
  sort(list.dirs(TEST_DATA_ROOT, recursive = FALSE, full.names = FALSE))
} else character(0)

existing_figs   <- if (dir.exists(FIGURES_PATH)) sort(list.files(FIGURES_PATH, "\\.png$")) else character(0)
existing_tables <- if (dir.exists(TABLES_PATH))  sort(list.files(TABLES_PATH,  "\\.csv$")) else character(0)

message("Loading 3D U-Net model...")
UNET_MODEL <- tryCatch({
  m <- jit_load(MODEL_PATH); m$to(device = "cpu"); m
}, error = function(e) { message("Model not found"); NULL })

REGION_PAL <- c(necrosis = "#ef4444", edema = "#f59e0b",
                enhancing = "#22c55e", label_4 = "#60a5fa")
SEG_PAL    <- c("0" = "#050a15", "1" = "#ef4444", "2" = "#f59e0b",
                "3" = "#22c55e", "4" = "#60a5fa")
SEG_LABS   <- c("0" = "Background", "1" = "Necrosis", "2" = "Edema",
                "3" = "Enhancing", "4" = "Enh. Core")
EVENT_PAL  <- c(CONTINUATION = "#22c55e", APPEARANCE = "#ef4444",
                DISAPPEARANCE = "#a855f7", SPLIT = "#f59e0b",
                MERGE = "#60a5fa")

dark_theme <- function(base_size = 13) {
  theme_minimal(base_size = base_size) %+replace% theme(
    plot.background   = element_rect(fill = "#050a15", colour = NA),
    panel.background  = element_rect(fill = "#050a15", colour = NA),
    text              = element_text(colour = "#c8ddf0"),
    axis.text         = element_text(colour = "#5a7a9a"),
    axis.title        = element_text(colour = "#8aaabe"),
    panel.grid.major  = element_line(colour = "#1a2a4a", linewidth = 0.3),
    panel.grid.minor  = element_blank(),
    legend.background = element_rect(fill = "#050a15", colour = NA),
    legend.text       = element_text(colour = "#c8ddf0"),
    legend.title      = element_text(colour = "#8aaabe"),
    strip.text        = element_text(colour = "#c8ddf0", face = "bold"),
    plot.title        = element_text(face = "bold", colour = "#ffffff",
                                     size = rel(1.15)),
    plot.subtitle     = element_text(colour = "#5a7a9a", size = rel(0.9))
  )
}

CSS <- "
body{font-family:'Segoe UI',system-ui,sans-serif;}

.stat-card{background:#0d1f3c;border:1px solid rgba(0,212,255,.12);
  border-radius:14px;padding:16px 12px;text-align:center;
  border-left:3px solid #0088ff;transition:transform .15s;}
.stat-card:hover{transform:translateY(-2px);
  box-shadow:0 0 15px rgba(0,136,255,.15);}
.stat-icon{font-size:20px;margin-bottom:4px;}
.stat-value{font-size:28px;font-weight:800;line-height:1.1;}
.stat-label{font-size:10px;color:#5a7a9a;margin-top:4px;
  text-transform:uppercase;letter-spacing:.8px;}
.clr-cr{color:#22c55e;text-shadow:0 0 12px rgba(34,197,94,.5);}
.clr-pr{color:#60a5fa;text-shadow:0 0 12px rgba(96,165,250,.5);}
.clr-sd{color:#f59e0b;text-shadow:0 0 12px rgba(245,158,11,.5);}
.clr-pd{color:#ef4444;text-shadow:0 0 12px rgba(239,68,68,.5);}

.card-box{background:#0d1f3c;border:1px solid rgba(0,212,255,.1);
  border-radius:14px;padding:20px;margin-bottom:16px;}
.card-box h4{margin-top:0;color:#fff;font-size:15px;font-weight:700;}

#preview_btn{background:rgba(0,212,255,.08);
  border:1px solid rgba(0,212,255,.3);border-radius:10px;
  color:#00d4ff;font-weight:600;padding:12px 0;width:100%;
  margin-bottom:8px;transition:all .15s;}
#preview_btn:hover{background:rgba(0,212,255,.15);
  box-shadow:0 0 12px rgba(0,212,255,.2);}
#run_btn{background:linear-gradient(135deg,#0066cc,#0088ff);
  border:none;border-radius:10px;color:#fff;font-weight:700;
  padding:14px 0;font-size:15px;width:100%;transition:all .15s;}
#run_btn:hover{transform:translateY(-1px);
  box-shadow:0 0 20px rgba(0,136,255,.45);}

.selectize-input{background:#0a1628!important;color:#c8ddf0!important;
  border:1px solid rgba(0,212,255,.2)!important;border-radius:10px!important;}
.selectize-dropdown{background:#0d1f3c!important;color:#c8ddf0!important;
  border:1px solid rgba(0,212,255,.2)!important;}
.selectize-dropdown-content .option{color:#c8ddf0!important;}
.selectize-dropdown-content .active{background:#0088ff!important;}

.nav-pills .nav-link{color:#5a7a9a;border-radius:10px;padding:7px 16px;
  font-weight:600;font-size:12px;}
.nav-pills .nav-link.active{background:#0088ff;color:#fff;
  box-shadow:0 0 10px rgba(0,136,255,.4);}

table.dataTable{color:#c8ddf0!important;}
table.dataTable thead th{color:#5a7a9a!important;
  border-bottom:1px solid rgba(0,212,255,.15)!important;}
table.dataTable tbody td{border-top:1px solid #0a1628!important;}
table.dataTable tbody tr{background:#0d1f3c!important;}
table.dataTable tbody tr:hover{background:#142a4a!important;}
.dataTables_wrapper .dataTables_filter input,
.dataTables_wrapper .dataTables_length select{
  background:#0a1628;color:#c8ddf0;border:1px solid rgba(0,212,255,.2);
  border-radius:6px;}
.dataTables_wrapper .dataTables_info,
.dataTables_wrapper .dataTables_paginate .paginate_button{
  color:#5a7a9a!important;}

.drop-zone .form-group{margin-bottom:0!important;}
.drop-zone .input-group .btn{display:none!important;}
.drop-zone .input-group{border:2px dashed rgba(0,212,255,.25);
  border-radius:12px;padding:14px;text-align:center;
  background:rgba(0,212,255,.03);transition:all .3s;cursor:pointer;}
.drop-zone .input-group:hover{border-color:rgba(0,212,255,.5);
  background:rgba(0,212,255,.06);}
.drop-zone .form-control{border:none!important;background:transparent!important;
  text-align:center;color:#5a7a9a!important;font-size:12px!important;}
.drop-zone label{display:none!important;}

.or-div{text-align:center;color:#3a5a7a;font-size:11px;
  margin:8px 0;letter-spacing:1px;}

/* radio buttons modern */
.radio-inline{margin-right:2px!important;}
.radio-inline label{color:#5a8aaa!important;font-size:11px!important;font-weight:600!important;
  padding:5px 10px!important;border-radius:8px!important;border:1px solid rgba(0,212,255,0.1)!important;
  transition:all 0.2s!important;cursor:pointer!important;background:rgba(0,212,255,0.03)!important;}
.radio-inline label:hover{border-color:rgba(0,212,255,0.3)!important;color:#00d4ff!important;
  background:rgba(0,212,255,0.06)!important;}
.radio-inline input[type=radio]:checked+span,
.radio-inline input:checked~*{color:#fff!important;}

/* sidebar button hover */
#preview_btn:hover{background:rgba(0,212,255,0.12)!important;
  border-color:rgba(0,212,255,0.5)!important;color:#00d4ff!important;
  box-shadow:0 0 15px rgba(0,212,255,0.15)!important;}
#run_btn:hover{transform:translateY(-1px)!important;
  box-shadow:0 4px 20px rgba(0,136,255,0.4)!important;
  background:linear-gradient(135deg,#0077dd,#00aaff)!important;}

.legend-strip{display:flex;gap:14px;justify-content:center;padding:8px 0;}
.legend-chip{display:flex;align-items:center;gap:5px;font-size:12px;color:#8aaabe;}
.legend-dot{width:12px;height:12px;border-radius:3px;display:inline-block;}

pre{background:#050a15!important;color:#7ab8e0!important;
  border:1px solid rgba(0,212,255,.1);border-radius:10px;padding:14px;}

.info-row{display:flex;justify-content:space-between;padding:3px 0;
  font-size:13px;border-bottom:1px solid rgba(0,212,255,.04);}
.info-label{color:#5a7a9a;}.info-value{color:#c8ddf0;font-weight:600;}

.welcome{text-align:center;padding:80px 40px;}
.welcome h3{color:#3a6a9a;font-weight:300;font-size:22px;}
.welcome p{color:#2a5a7a;}

.form-label,.control-label{color:#5a7a9a!important;}
"

desc_p <- function(txt) {
  tags$p(style = "color:#5a7a9a;font-size:13px;line-height:1.5;margin-bottom:12px;",
         HTML(txt))
}
card_div <- function(...) {
  div(class = "card-box", ...)
}

# ══════════════════════════════════════════════════════════════
#                              UI
# ══════════════════════════════════════════════════════════════

ui <- page_sidebar(
  title = NULL,
  theme = bs_theme(version = 5, bg = "#050a15", fg = "#c8ddf0",
                   primary = "#0088ff", secondary = "#0a1628",
                   font_scale = 0.95, "card-bg" = "#0d1f3c"),
  tags$head(tags$style(HTML(CSS))),

  sidebar = sidebar(
    width = 290, bg = "#0a1228", open = TRUE,
    div(style = "padding:12px 8px 0;",

      # ── Logo & Title ──
      div(style = "text-align:center;margin-bottom:16px;padding:14px 0;
          background:linear-gradient(135deg,rgba(0,136,255,0.08),rgba(0,212,255,0.04));
          border-radius:14px;border:1px solid rgba(0,212,255,0.1);",
        tags$span(style = "font-size:32px;display:block;
            filter:drop-shadow(0 0 8px rgba(0,212,255,0.4));", "\U0001F9E0"),
        div(style = "font-size:16px;font-weight:800;color:#fff;margin-top:6px;
            letter-spacing:0.5px;text-shadow:0 0 15px rgba(0,212,255,0.3);",
            "Tumor Progression"),
        div(style = "font-size:10px;color:#4a7a9a;margin-top:2px;
            letter-spacing:1.5px;text-transform:uppercase;",
            "Region-Wise Analysis")
      ),

      # ── Data Source ──
      div(style = "margin-bottom:12px;",
        div(style = "color:#4a7a9a;font-size:9px;letter-spacing:1.5px;
            text-transform:uppercase;margin-bottom:6px;font-weight:600;",
            "\U0001F4C1 DATA SOURCE"),
        radioButtons("data_source", NULL,
                      choices = c("LUMIERE Dataset" = "lumiere",
                                  "Test Data" = "test",
                                  "Custom Path" = "custom"),
                      selected = "lumiere", inline = TRUE)
      ),

      # ── Patient Selection ──
      div(style = "margin-bottom:14px;",
        div(style = "color:#4a7a9a;font-size:9px;letter-spacing:1.5px;
            text-transform:uppercase;margin-bottom:6px;font-weight:600;",
            "\U0001F50D SELECT PATIENT"),
        conditionalPanel(
          condition = "input.data_source == 'lumiere'",
          selectizeInput("patient_id", NULL, choices = patient_list,
                         selected = "Patient-083",
                         options = list(placeholder = "Search patient..."))
        ),
        conditionalPanel(
          condition = "input.data_source == 'test'",
          selectizeInput("test_patient_id", NULL, choices = test_list,
                         selected = if (length(test_list) > 0) test_list[1] else NULL,
                         options = list(placeholder = "Search test patient..."))
        ),
        conditionalPanel(
          condition = "input.data_source == 'custom'",
          textInput("custom_path", NULL,
                    placeholder = "C:/path/to/Patient-XXX")
        )
      ),

      # ── Action Buttons ──
      div(style = "display:flex;flex-direction:column;gap:8px;margin-bottom:14px;",
        actionButton("preview_btn", "\U0001F441 Preview MRI",
                      style = "width:100%;padding:11px;border-radius:10px;font-weight:600;
                      font-size:13px;background:rgba(0,212,255,0.06);
                      border:1px solid rgba(0,212,255,0.25);color:#7ac0e0;
                      transition:all 0.2s;"),
        actionButton("run_btn", "\U0001F680 Run Full Analysis",
                      style = "width:100%;padding:13px;border-radius:10px;font-weight:700;
                      font-size:14px;background:linear-gradient(135deg,#0055cc,#0088ff);
                      border:none;color:#fff;letter-spacing:0.3px;
                      transition:all 0.2s;box-shadow:0 2px 12px rgba(0,136,255,0.25);")
      ),

      # ── Status ──
      htmlOutput("status_text"),

      # ── Patient Info ──
      div(style = "margin-top:12px;padding:10px;background:rgba(0,212,255,0.03);
          border-radius:10px;border:1px solid rgba(0,212,255,0.08);",
        htmlOutput("patient_info")
      ),

      # ── Footer ──
      div(style = "text-align:center;margin-top:16px;padding-top:12px;
          border-top:1px solid rgba(0,212,255,0.08);",
        div(style = "font-size:9px;color:#2a4a6a;letter-spacing:1px;text-transform:uppercase;",
            "VIT Chennai \U00B7 MTech ISE"),
        div(style = "font-size:8px;color:#1a3a5a;margin-top:2px;",
            "3D U-Net + LSTM \U00B7 LUMIERE Dataset")
      )
    )
  ),

  # stat cards
  layout_columns(
    col_widths = c(3, 3, 3, 3), fill = FALSE,
    div(class = "stat-card", htmlOutput("stat_prog")),
    div(class = "stat-card", htmlOutput("stat_regions")),
    div(class = "stat-card", htmlOutput("stat_events")),
    div(class = "stat-card", htmlOutput("stat_conf"))
  ),

  navset_pill(
    id = "tabs",

    # ── Patient Summary ──
    nav_panel("Patient Summary", icon = icon("user-injured"), br(),
      htmlOutput("clinical_summary")
    ),

    # ── MRI Preview ──
    nav_panel("MRI Preview", icon = icon("eye"), br(),
      card_div(
        h4("Raw MRI \u2014 Full Resolution"),
        desc_p(paste0(
          "Full-resolution brain scans. Use the slider to scroll through ",
          "axial slices. <b>T1</b> = anatomy, <b>T1c</b> = contrast ",
          "(tumour lights up), <b>T2</b> = fluid bright, ",
          "<b>FLAIR</b> = edema visible.")),
        fluidRow(
          column(4, selectInput("pv_tp", "TIMEPOINT", choices = NULL)),
          column(4, sliderInput("pv_slice", "AXIAL SLICE", 1, 100, 50)),
          column(4, htmlOutput("pv_info"))
        ),
        plotOutput("mri_preview_plot", height = "380px")
      )
    ),

    # ── MRI Scans (preprocessed) ──
    nav_panel("MRI Scans", icon = icon("brain"), br(),
      card_div(
        h4("Preprocessed Multi-Modal MRI"),
        desc_p("Each column is a different MRI modality (48\u00B3). Rows = timepoints."),
        plotOutput("mri_grid", height = "480px")
      )
    ),

    # ── Segmentation ──
    nav_panel("Segmentation", icon = icon("crosshairs"), br(),
      desc_p(paste0(
        "The 3D U-Net predicts tumour regions per voxel. ",
        "<b>Left:</b> raw mask. <b>Right:</b> overlay on FLAIR.")),
      layout_columns(
        col_widths = c(6, 6),
        card_div(h4("Segmentation Mask"),
                 plotOutput("seg_plot", height = "400px")),
        card_div(h4("Overlay on FLAIR"),
                 plotOutput("overlay_plot", height = "400px"))
      ),
      div(class = "legend-strip",
        div(class="legend-chip",div(class="legend-dot",style="background:#ef4444;"),"Necrosis (dead core)"),
        div(class="legend-chip",div(class="legend-dot",style="background:#f59e0b;"),"Edema (swelling)"),
        div(class="legend-chip",div(class="legend-dot",style="background:#22c55e;"),"Enhancing (active)"),
        div(class="legend-chip",div(class="legend-dot",style="background:#60a5fa;"),"Enh. Core")
      )
    ),

    # ── Regions ──
    nav_panel("Regions", icon = icon("chart-bar"), br(),
      desc_p("Each connected blob of tumour voxels is a region with measured volume."),
      layout_columns(
        col_widths = c(7, 5),
        card_div(h4("Volume by Region & Timepoint"),
                 plotOutput("volume_bar", height = "360px")),
        card_div(h4("Volume Distribution"),
                 plotOutput("volume_box", height = "360px"))
      ),
      card_div(h4("Region Detail"), DTOutput("region_dt"))
    ),

    # ── Tracking ──
    nav_panel("Tracking", icon = icon("route"), br(),
      desc_p(paste0(
        "Regions matched across timepoints. ",
        "<b>CONTINUATION</b>=persists, <b>APPEARANCE</b>=new, ",
        "<b>DISAPPEARANCE</b>=gone, <b>SPLIT/MERGE</b>.")),
      layout_columns(
        col_widths = c(5, 7),
        card_div(h4("Event Summary"),
                 plotOutput("event_bar", height = "320px")),
        card_div(h4("Event Detail"), DTOutput("event_dt"))
      )
    ),

    # ── Progression ──
    nav_panel("Progression", icon = icon("chart-line"), br(),
      desc_p(paste0(
        "RANO: <b>CR</b> = all gone, <b>PR</b> = shrinking, ",
        "<b>SD</b> = stable, <b>PD</b> = growing/new lesions.")),
      card_div(plotOutput("trajectory_plot", height = "360px")),
      layout_columns(
        col_widths = c(6, 6),
        card_div(htmlOutput("progression_card")),
        card_div(h4("Trajectory Detail"), DTOutput("traj_dt"))
      )
    ),

    # ── Report ──
    nav_panel("Report", icon = icon("file-medical"), br(),
      card_div(
        h4("Full Patient Report"),
        desc_p("Complete text report. Copy for documentation."),
        verbatimTextOutput("full_report")
      )
    ),

    # ── Pipeline Results ──
    nav_panel("Pipeline Results", icon = icon("images"), br(),
      layout_columns(
        col_widths = c(8, 4),
        card_div(
          h4("Generated Figures"),
          selectInput("fig_sel", NULL, choices = existing_figs),
          imageOutput("fig_display", height = "auto")
        ),
        card_div(
          h4("Result Tables"),
          tags$ul(style = "list-style:none;padding:0;",
            lapply(existing_tables, function(f) {
              tags$li(style = "padding:4px 0;border-bottom:1px solid rgba(0,212,255,.05);",
                tags$a(href = file.path("results/tables", f), f,
                       style = "color:#00d4ff;font-size:12px;",
                       download = f))
            })
          )
        )
      )
    )
  ),

  # welcome
  conditionalPanel(
    "!output.has_results && !output.has_preview",
    div(class = "welcome",
      tags$span(style = "font-size:52px;opacity:.3;", "\U0001F9E0"),
      h3("Select a patient and click Preview or Analyse"),
      p("Load MRI \u2192 3D U-Net \u2192 region tracking \u2192 RANO")
    )
  )
)

# ══════════════════════════════════════════════════════════════
#                           SERVER
# ══════════════════════════════════════════════════════════════

server <- function(input, output, session) {

  addResourcePath("results", file.path(PROJECT_ROOT, "results"))

  rv <- reactiveValues(
    status = "", preview = NULL, patient_data = NULL,
    preprocessed = NULL, segmented = NULL,
    region_data = NULL, events = NULL, trajectories = NULL,
    progression = NULL, results = NULL,
    upload_dir = NULL, all_wdirs = NULL
  )

  output$has_results <- reactive(!is.null(rv$results))
  outputOptions(output, "has_results", suspendWhenHidden = FALSE)
  output$has_preview <- reactive(!is.null(rv$preview))
  outputOptions(output, "has_preview", suspendWhenHidden = FALSE)

  # file upload
  observeEvent(input$upload_files, {
    files <- input$upload_files
    if (is.null(files) || nrow(files) == 0) return()
    tmp <- file.path(tempdir(), "uploaded_patient", "week-000")
    dir.create(tmp, recursive = TRUE, showWarnings = FALSE)
    for (i in seq_len(nrow(files)))
      file.copy(files$datapath[i], file.path(tmp, files$name[i]),
                overwrite = TRUE)
    rv$upload_dir <- dirname(tmp)
    rv$status <- paste0(nrow(files), " file(s) ready")
  })

  get_path <- function() {
    if (!is.null(rv$upload_dir)) return(rv$upload_dir)
    if (input$data_source == "custom" && nzchar(trimws(input$custom_path))) {
      return(normalizePath(trimws(input$custom_path), winslash = "/", mustWork = FALSE))
    }
    if (input$data_source == "test") {
      return(file.path(TEST_DATA_ROOT, input$test_patient_id))
    }
    file.path(IMAGING_ROOT, input$patient_id)
  }
  get_pid <- function() {
    if (!is.null(rv$upload_dir)) return("Uploaded-Patient")
    if (input$data_source == "custom" && nzchar(trimws(input$custom_path))) {
      return(basename(trimws(input$custom_path)))
    }
    if (input$data_source == "test") return(input$test_patient_id)
    input$patient_id
  }

  # ── Preview MRI ──

  load_timepoint <- function(wd) {
    vols <- list()
    atlas <- file.path(wd, "DeepBraTumIA-segmentation", "atlas", "skull_strip")
    if (dir.exists(atlas)) {
      paths <- list(
        T1   = file.path(atlas, "t1_skull_strip.nii.gz"),
        T1c  = file.path(atlas, "ct1_skull_strip.nii.gz"),
        T2   = file.path(atlas, "t2_skull_strip.nii.gz"),
        FLAIR = file.path(atlas, "flair_skull_strip.nii.gz"))
    } else {
      paths <- list(
        T1   = file.path(wd, "T1.nii.gz"),
        T1c  = file.path(wd, "CT1.nii.gz"),
        T2   = file.path(wd, "T2.nii.gz"),
        FLAIR = file.path(wd, "FLAIR.nii.gz"))
    }
    for (m in names(paths)) {
      if (file.exists(paths[[m]])) {
        vols[[m]] <- tryCatch(as.array(readNifti(paths[[m]])),
                              error = function(e) NULL)
        gc()
      }
    }
    seg_p <- file.path(wd, "DeepBraTumIA-segmentation",
                       "atlas", "segmentation", "seg_mask.nii.gz")
    seg_v <- if (file.exists(seg_p)) {
      tryCatch(as.array(readNifti(seg_p)), error = function(e) NULL)
    } else NULL
    if (length(vols) == 0) return(NULL)
    list(volumes = vols, seg = seg_v, dims = dim(vols[[1]]))
  }

  observeEvent(input$preview_btn, {
    pp <- get_path()
    if (!dir.exists(pp)) { rv$status <- "Folder not found"; return() }
    rv$status <- "Loading MRI preview..."

    all_wdirs <- sort(list.dirs(pp, recursive = FALSE, full.names = TRUE))
    all_wdirs <- all_wdirs[grepl("week-", basename(all_wdirs))]
    if (length(all_wdirs) == 0) { rv$status <- "No timepoints found"; return() }

    rv$all_wdirs <- all_wdirs

    # Load only first timepoint initially (fast)
    rv$status <- "Loading first timepoint..."
    first_tp <- load_timepoint(all_wdirs[1])
    if (is.null(first_tp)) { rv$status <- "Could not load MRI"; return() }

    prev <- list()
    prev[[basename(all_wdirs[1])]] <- first_tp

    if (length(prev) == 0) { rv$status <- "Could not load MRI"; return() }
    rv$preview <- prev
    tps <- basename(all_wdirs)
    updateSelectInput(session, "pv_tp", choices = tps, selected = tps[1])
    d <- prev[[1]]$dims
    updateSliderInput(session, "pv_slice", min = 1, max = d[3],
                      value = d[3] %/% 2)
    rv$status <- paste("Preview loaded. Select timepoint to view others.")
    tryCatch(nav_select("tabs", "MRI Preview"), error = function(e) NULL)
  })

  observeEvent(input$pv_tp, {
    req(input$pv_tp, rv$all_wdirs)
    # Lazy load: if timepoint not cached, load it now
    if (is.null(rv$preview[[input$pv_tp]])) {
      wd <- rv$all_wdirs[basename(rv$all_wdirs) == input$pv_tp]
      if (length(wd) == 1) {
        rv$status <- paste("Loading", input$pv_tp, "...")
        tp_data <- load_timepoint(wd)
        if (!is.null(tp_data)) {
          rv$preview[[input$pv_tp]] <- tp_data
          rv$status <- paste("Loaded", input$pv_tp)
        }
      }
    }
    tp <- rv$preview[[input$pv_tp]]
    if (!is.null(tp))
      updateSliderInput(session, "pv_slice", min = 1, max = tp$dims[3],
                        value = tp$dims[3] %/% 2)
  })

  output$pv_info <- renderUI({
    req(rv$preview, input$pv_tp)
    tp <- rv$preview[[input$pv_tp]]
    if (is.null(tp)) return(NULL)
    d <- tp$dims
    HTML(paste0(
      "<div style='padding-top:6px;'>",
      "<div style='color:#5a7a9a;font-size:10px;text-transform:uppercase;'>Volume</div>",
      "<div style='color:#c8ddf0;font-weight:700;'>",
      d[1], "x", d[2], "x", d[3], "</div>",
      "<div style='color:#8aaabe;font-size:11px;'>",
      paste(names(tp$volumes), collapse = ", "), "</div>",
      "<div style='color:#8aaabe;font-size:11px;'>Seg: ",
      if (!is.null(tp$seg))
        "<span style='color:#22c55e;'>Yes</span>"
      else "<span style='color:#f59e0b;'>No</span>",
      "</div></div>"))
  })

  output$mri_preview_plot <- renderPlot({
    req(rv$preview, input$pv_tp, input$pv_slice)
    tp <- rv$preview[[input$pv_tp]]
    if (is.null(tp)) return(NULL)
    z <- max(1L, min(as.integer(input$pv_slice), tp$dims[3]))
    plots <- list()

    for (m in names(tp$volumes)) {
      v <- tp$volumes[[m]]
      if (is.null(v)) next
      sl <- v[, , z]
      df <- expand.grid(x = seq_len(nrow(sl)), y = seq_len(ncol(sl)))
      df$v <- as.vector(sl)
      plots[[m]] <- ggplot(df, aes(x, y, fill = v)) +
        geom_raster() +
        scale_fill_viridis_c(option = "inferno", guide = "none") +
        coord_fixed() + labs(title = m) +
        theme_void() +
        theme(plot.title = element_text(hjust = .5, size = 11,
                color = "#c8ddf0", face = "bold"),
              plot.background = element_rect(fill = "#050a15", colour = NA))
    }
    if (!is.null(tp$seg)) {
      sl <- tp$seg[, , min(z, dim(tp$seg)[3])]
      df <- expand.grid(x = seq_len(nrow(sl)), y = seq_len(ncol(sl)))
      df$label <- factor(as.vector(sl))
      plots[["Segmentation"]] <- ggplot(df, aes(x, y, fill = label)) +
        geom_raster() +
        scale_fill_manual(values = SEG_PAL, guide = "none") +
        coord_fixed() + labs(title = "Segmentation") +
        theme_void() +
        theme(plot.title = element_text(hjust = .5, size = 11,
                color = "#c8ddf0", face = "bold"),
              plot.background = element_rect(fill = "#050a15", colour = NA))
    }
    if (length(plots) == 0) return(NULL)
    wrap_plots(plots, nrow = 1) +
      plot_annotation(theme = theme(
        plot.background = element_rect(fill = "#050a15", colour = NA)))
  }, bg = "#050a15")

  # ── Full Analysis ──

  observeEvent(input$run_btn, {
    pp <- get_path()
    pid <- get_pid()
    if (!dir.exists(pp)) { rv$status <- "Not found"; return() }
    if (is.null(UNET_MODEL)) { rv$status <- "Model not loaded"; return() }

    rv$results <- NULL
    withProgress(message = paste("Analysing", pid), value = 0, {

      incProgress(.10, detail = "Loading MRI...")
      rv$status <- "Loading..."
      pd <- tryCatch(load_full_patient(pp),
                     error = function(e) { rv$status <- e$message; NULL })
      if (is.null(pd)) return()
      rv$patient_data <- pd

      incProgress(.20, detail = "Preprocessing...")
      rv$status <- "Preprocessing..."
      preprocessed <- tryCatch(
        preprocess_full_patient(pd, target_size = TARGET_SIZE),
        error = function(e) { rv$status <- e$message; NULL })
      if (is.null(preprocessed)) return()
      rv$preprocessed <- preprocessed

      incProgress(.20, detail = "U-Net segmentation...")
      rv$status <- "3D U-Net segmentation..."
      segmented <- segment_patient(UNET_MODEL, preprocessed, "cpu")
      rv$segmented <- segmented

      incProgress(.15, detail = "Extracting regions...")
      rd <- extract_all_patient_regions(segmented, pid, min_vol = 50)
      rv$region_data <- rd

      incProgress(.10, detail = "Tracking...")
      events <- NULL; trajectories <- NULL; progression <- NULL
      if (length(rd$regions_by_timepoint) >= 2) {
        tracking <- track_patient_regions(rd$regions_by_timepoint)
        events <- create_event_table(tracking$all_events, pid)
        tp_ns <- names(rd$regions_by_timepoint)
        tw <- sapply(pd$timepoints[tp_ns], `[[`, "week_number")
        trajectories <- list()
        for (r in rd$regions_by_timepoint[[1]]) {
          vols <- sapply(tp_ns, function(tp) {
            mt <- Filter(function(x) x$region_type == r$region_type,
                         rd$regions_by_timepoint[[tp]])
            if (length(mt) > 0) mt[[1]]$volume_cm3 else 0
          })
          trajectories[[r$region_id]] <-
            analyze_region_trajectory(vols, tw)
        }
        if (is.null(events)) events <- tibble(event_type = character(0))
        progression <- classify_patient_progression(trajectories, events)
      }
      rv$events <- events
      rv$trajectories <- trajectories
      rv$progression <- progression

      incProgress(.15, detail = "Done!")
      rv$results <- list(pid = pid, preprocessed = preprocessed,
                         segmented = segmented, region_table = rd$table)
      rv$status <- paste("\u2705 Analysis complete:", pid)
    })
  })

  # ── Sidebar outputs ──

  output$status_text <- renderUI({
    HTML(paste0("<div style='color:#5a7a9a;font-size:12px;min-height:18px;'>",
                rv$status, "</div>"))
  })

  output$patient_info <- renderUI({
    pd <- rv$patient_data
    if (is.null(pd)) return(NULL)
    tps <- names(pd$timepoints)
    row <- function(l, v) paste0(
      "<div class='info-row'><span class='info-label'>", l,
      "</span><span class='info-value'>", v, "</span></div>")
    HTML(paste0(row("Patient", pd$patient_id),
                row("Timepoints", length(tps)),
                row("Scans", paste(tps, collapse = ", "))))
  })

  # ── Stat cards ──

  output$stat_prog <- renderUI({
    p <- rv$progression
    if (is.null(p)) return(HTML(
      "<div class='stat-icon'>\U0001F4CA</div>",
      "<div class='stat-value' style='color:#3a5a7a;'>--</div>",
      "<div class='stat-label'>Classification</div>"))
    cls <- p$classification
    lab <- switch(cls, CR = "Complete Response", PR = "Partial Response",
                  SD = "Stable Disease", PD = "Progressive Disease", cls)
    HTML(paste0(
      "<div class='stat-icon'>\U0001F3AF</div>",
      "<div class='stat-value clr-", tolower(cls), "'>", cls, "</div>",
      "<div class='stat-label'>", lab, "</div>"))
  })
  output$stat_regions <- renderUI({
    n <- if (!is.null(rv$region_data)) nrow(rv$region_data$table) else "--"
    HTML(paste0(
      "<div class='stat-icon'>\U0001F9EC</div>",
      "<div class='stat-value'>", n, "</div>",
      "<div class='stat-label'>Regions</div>"))
  })
  output$stat_events <- renderUI({
    n <- if (!is.null(rv$events)) nrow(rv$events) else "--"
    HTML(paste0(
      "<div class='stat-icon'>\U0001F504</div>",
      "<div class='stat-value'>", n, "</div>",
      "<div class='stat-label'>Events</div>"))
  })
  output$stat_conf <- renderUI({
    p <- rv$progression
    v <- if (!is.null(p)) paste0(round(p$confidence * 100), "%") else "--"
    HTML(paste0(
      "<div class='stat-icon'>\U0001F4C8</div>",
      "<div class='stat-value'>", v, "</div>",
      "<div class='stat-label'>Confidence</div>"))
  })

  # ── Patient Summary ──

  output$clinical_summary <- renderUI({
    req(rv$results, rv$progression)
    pd <- rv$patient_data; rt <- rv$region_data$table
    prog <- rv$progression; ev <- rv$events; trajs <- rv$trajectories
    tps <- names(pd$timepoints)
    wks <- sapply(pd$timepoints, `[[`, "week_number")
    first_v <- rt %>% filter(timepoint == min(timepoint)) %>%
      summarise(v = sum(volume_cm3, na.rm = TRUE)) %>% pull(v)
    last_v  <- rt %>% filter(timepoint == max(timepoint)) %>%
      summarise(v = sum(volume_cm3, na.rm = TRUE)) %>% pull(v)
    pct <- if (first_v > 0) round((last_v - first_v)/first_v*100, 1) else 0
    arrow <- if (pct > 5) "\U0001F53A" else if (pct < -5) "\U0001F53B" else "\u27A1"
    cls <- prog$classification
    clr <- switch(cls, CR="#22c55e", PR="#60a5fa", SD="#f59e0b", PD="#ef4444", "#5a7a9a")
    lbl <- switch(cls,
      CR = "Complete Response \u2014 All tumour gone.",
      PR = "Partial Response \u2014 Shrinking.",
      SD = "Stable Disease \u2014 No significant change.",
      PD = "Progressive Disease \u2014 Growing or new lesions.", cls)

    largest <- rt %>% arrange(desc(volume_cm3)) %>% slice(1)
    type_tbl <- rt %>% group_by(region_type) %>%
      summarise(Count = n(), `Avg Vol` = round(mean(volume_cm3, na.rm=TRUE), 2),
                `Max Vol` = round(max(volume_cm3, na.rm=TRUE), 2), .groups = "drop")
    type_rows <- paste0(apply(type_tbl, 1, function(r) {
      col <- switch(r[1], necrosis="#ef4444", edema="#f59e0b",
                    enhancing="#22c55e", "#60a5fa")
      paste0("<tr><td style='color:", col, ";font-weight:700;'>",
             toupper(r[1]), "</td><td>", r[2], "</td><td>",
             r[3], " cm\u00B3</td><td>", r[4], " cm\u00B3</td></tr>")
    }), collapse = "")

    traj_rows <- if (!is.null(trajs)) paste0(sapply(names(trajs), function(rid) {
      t <- trajs[[rid]]
      tc <- switch(t$trend, GROWTH="#ef4444", SHRINKAGE="#22c55e",
                   STABLE="#f59e0b", DISAPPEARED="#a855f7", "#5a7a9a")
      pc <- if (!is.na(t$percent_change)) sprintf("%+.0f%%", t$percent_change*100) else "N/A"
      paste0("<tr><td>", rid, "</td><td style='color:", tc, ";font-weight:700;'>",
             t$trend, "</td><td>", round(t$first_volume, 1),
             "</td><td>", round(t$last_volume, 1), "</td><td>", pc, "</td></tr>")
    }), collapse = "") else ""

    n_ev <- if (!is.null(ev)) nrow(ev) else 0
    n_dis <- if (!is.null(ev)) sum(ev$event_type == "DISAPPEARANCE") else 0
    n_app <- if (!is.null(ev)) sum(ev$event_type == "APPEARANCE") else 0

    HTML(paste0(
      "<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:16px;'>",
      "<div class='card-box' style='padding:14px;margin:0;border-left:3px solid #0088ff;'>",
      "<div style='color:#5a7a9a;font-size:10px;text-transform:uppercase;'>Scan Timeline</div>",
      "<div style='color:#fff;font-weight:700;'>", length(tps), " scans \u2014 Week ",
      min(wks), " to ", max(wks), "</div>",
      "<div style='color:#8aaabe;font-size:11px;'>",
      paste(tps, collapse = " \u2192 "), "</div></div>",

      "<div class='card-box' style='padding:14px;margin:0;border-left:3px solid #0088ff;'>",
      "<div style='color:#5a7a9a;font-size:10px;text-transform:uppercase;'>Total Tumour Volume</div>",
      "<div style='color:#fff;font-weight:700;'>",
      round(first_v, 1), " \u2192 ", round(last_v, 1), " cm\u00B3 ", arrow, "</div>",
      "<div style='color:#8aaabe;font-size:11px;'>Change: ",
      sprintf("%+.1f%%", pct), "</div></div>",

      "<div class='card-box' style='padding:14px;margin:0;border-left:3px solid ", clr, ";'>",
      "<div style='color:#5a7a9a;font-size:10px;text-transform:uppercase;'>RANO Classification</div>",
      "<div style='color:", clr, ";font-size:22px;font-weight:800;",
      "text-shadow:0 0 15px ", clr, ";'>", cls, "</div>",
      "<div style='color:#8aaabe;font-size:11px;'>Confidence: ",
      round(prog$confidence*100), "%</div></div></div>",

      "<div class='card-box'><h4>\U0001F4CB Interpretation</h4>",
      "<p style='color:#c8ddf0;font-size:14px;'><span style='color:", clr,
      ";font-weight:700;font-size:16px;'>", cls, "</span> \u2014 ", lbl, "</p></div>",

      "<div class='card-box'><h4>\U0001F3AF Tumour Regions (", nrow(rt), " total)</h4>",
      "<div style='color:#5a7a9a;font-size:12px;margin-bottom:8px;'>",
      "Largest: ", toupper(largest$region_type), " at ",
      round(largest$volume_cm3, 1), " cm\u00B3 (", largest$timepoint, ")</div>",
      "<table style='width:100%;color:#c8ddf0;font-size:13px;'>",
      "<tr style='color:#5a7a9a;'><th>Type</th><th>Count</th><th>Avg Vol</th><th>Max Vol</th></tr>",
      type_rows, "</table></div>",

      if (nchar(traj_rows) > 0) paste0(
        "<div class='card-box'><h4>\U0001F4C8 Region Trajectories</h4>",
        "<table style='width:100%;color:#c8ddf0;font-size:13px;'>",
        "<tr style='color:#5a7a9a;'><th>Region</th><th>Trend</th>",
        "<th>First</th><th>Last</th><th>Change</th></tr>",
        traj_rows, "</table></div>") else "",

      "<div class='card-box'><h4>\U0001F504 Events (", n_ev, " total)</h4>",
      "<p style='color:#8aaabe;font-size:13px;'>",
      if (n_dis > 0) paste0("<span style='color:#22c55e;'>\u2705 ",
                            n_dis, " disappeared</span> ") else "",
      if (n_app > 0) paste0("<span style='color:#ef4444;'>\u26A0 ",
                            n_app, " appeared</span>") else "",
      "</p></div>"
    ))
  })

  # ── MRI Scans (preprocessed grid) ──

  output$mri_grid <- renderPlot({
    req(rv$results)
    pp <- rv$results$preprocessed
    mod_nm <- c("T1", "T1c", "T2", "FLAIR")
    plots <- list()
    for (tp in names(pp)[seq_len(min(length(pp), 4))]) {
      stk <- pp[[tp]]$stacked
      if (is.null(stk)) next
      for (ch in seq_len(min(dim(stk)[4], 4))) {
        sl <- stk[, , dim(stk)[3] %/% 2, ch]
        df <- expand.grid(x = seq_len(nrow(sl)), y = seq_len(ncol(sl)))
        df$v <- as.vector(sl)
        ml <- if (ch <= 4) mod_nm[ch] else paste0("Ch", ch)
        plots[[length(plots)+1]] <- ggplot(df, aes(x, y, fill = v)) +
          geom_raster() +
          scale_fill_viridis_c(option = "inferno", guide = "none") +
          coord_fixed() + labs(title = paste(tp, "\u2014", ml)) +
          theme_void() +
          theme(plot.title = element_text(hjust = .5, size = 9,
                  color = "#7a9aba", face = "bold"),
                plot.background = element_rect(fill = "#050a15", colour = NA))
      }
    }
    if (length(plots) == 0) return(NULL)
    ncols <- min(dim(pp[[1]]$stacked)[4], 4)
    wrap_plots(plots, ncol = ncols) +
      plot_annotation(theme = theme(
        plot.background = element_rect(fill = "#050a15", colour = NA)))
  }, bg = "#050a15")

  # ── Segmentation mask ──

  output$seg_plot <- renderPlot({
    req(rv$results)
    seg_d <- rv$results$segmented
    plots <- list()
    for (tp in names(seg_d)[seq_len(min(length(seg_d), 4))]) {
      seg <- seg_d[[tp]]$segmentation
      if (is.null(seg)) next
      sl <- seg[, , dim(seg)[3] %/% 2]
      df <- expand.grid(x = seq_len(nrow(sl)), y = seq_len(ncol(sl)))
      df$label <- factor(as.vector(sl))
      plots[[tp]] <- ggplot(df, aes(x, y, fill = label)) +
        geom_raster() +
        scale_fill_manual(values = SEG_PAL, guide = "none") +
        coord_fixed() + labs(title = tp) +
        theme_void() +
        theme(plot.title = element_text(hjust = .5, size = 11,
                color = "#c8ddf0", face = "bold"),
              plot.background = element_rect(fill = "#050a15", colour = NA))
    }
    if (length(plots) == 0) return(NULL)
    wrap_plots(plots, nrow = 1) +
      plot_annotation(theme = theme(
        plot.background = element_rect(fill = "#050a15", colour = NA)))
  }, bg = "#050a15")

  # ── Overlay on FLAIR ──

  output$overlay_plot <- renderPlot({
    req(rv$results)
    seg_d <- rv$results$segmented
    pp <- rv$results$preprocessed
    plots <- list()
    seg_rgb <- list("1"=c(239,68,68), "2"=c(245,158,11),
                    "3"=c(34,197,94), "4"=c(96,165,250))
    for (tp in names(seg_d)[seq_len(min(length(seg_d), 4))]) {
      seg <- seg_d[[tp]]$segmentation
      stk <- pp[[tp]]$stacked
      if (is.null(seg) || is.null(stk)) next
      z <- dim(seg)[3] %/% 2
      flair_ch <- min(dim(stk)[4], 4)
      fl <- stk[, , z, flair_ch]
      fl <- (fl - min(fl)) / max(1e-8, max(fl) - min(fl))
      nr <- nrow(fl); nc <- ncol(fl)
      r <- fl; g <- fl; b <- fl
      sg <- seg[, , z]
      for (lbl in names(seg_rgb)) {
        mask <- sg == as.integer(lbl)
        if (!any(mask)) next
        clr <- seg_rgb[[lbl]] / 255
        alpha <- 0.5
        r[mask] <- r[mask] * (1 - alpha) + clr[1] * alpha
        g[mask] <- g[mask] * (1 - alpha) + clr[2] * alpha
        b[mask] <- b[mask] * (1 - alpha) + clr[3] * alpha
      }
      df <- expand.grid(x = seq_len(nr), y = seq_len(nc))
      df$col <- rgb(pmin(as.vector(r), 1),
                    pmin(as.vector(g), 1),
                    pmin(as.vector(b), 1))
      plots[[tp]] <- ggplot(df, aes(x, y, fill = col)) +
        geom_raster() +
        scale_fill_identity() +
        coord_fixed() + labs(title = tp) +
        theme_void() +
        theme(plot.title = element_text(hjust = .5, size = 11,
                color = "#c8ddf0", face = "bold"),
              plot.background = element_rect(fill = "#050a15", colour = NA))
    }
    if (length(plots) == 0) return(NULL)
    wrap_plots(plots, nrow = 1) +
      plot_annotation(theme = theme(
        plot.background = element_rect(fill = "#050a15", colour = NA)))
  }, bg = "#050a15")

  # ── Regions ──

  output$volume_bar <- renderPlot({
    req(rv$results)
    rt <- rv$results$region_table
    if (nrow(rt) == 0) return(NULL)
    ggplot(rt, aes(timepoint, volume_cm3, fill = region_type)) +
      geom_col(position = "dodge", alpha = .85, width = .6) +
      scale_fill_manual(values = REGION_PAL, name = "Region") +
      labs(title = "Region Volumes", x = "Timepoint",
           y = "Volume (cm\u00B3)") + dark_theme()
  }, bg = "#050a15")

  output$volume_box <- renderPlot({
    req(rv$results)
    rt <- rv$results$region_table
    if (nrow(rt) == 0) return(NULL)
    ggplot(rt, aes(region_type, volume_cm3, fill = region_type)) +
      geom_boxplot(alpha = .7) + geom_jitter(alpha = .4, width = .15) +
      scale_fill_manual(values = REGION_PAL, guide = "none") +
      labs(title = "Volume Distribution", x = "", y = "cm\u00B3") +
      dark_theme()
  }, bg = "#050a15")

  output$region_dt <- renderDT({
    req(rv$results)
    rv$results$region_table %>%
      select(timepoint, region_id, region_type, volume_cm3, n_voxels,
             centroid_x, centroid_y, centroid_z) %>%
      mutate(across(where(is.numeric), ~round(., 2))) %>%
      datatable(options = list(pageLength = 10, dom = "ftp"),
                rownames = FALSE, class = "stripe")
  })

  # ── Tracking ──

  output$event_bar <- renderPlot({
    req(rv$events)
    if (nrow(rv$events) == 0) return(NULL)
    rv$events %>% count(event_type) %>%
      mutate(event_type = fct_reorder(event_type, n, .desc = TRUE)) %>%
      ggplot(aes(event_type, n, fill = event_type)) +
      geom_col(alpha = .85) +
      scale_fill_manual(values = EVENT_PAL, guide = "none") +
      labs(title = "Events", x = "", y = "Count") + dark_theme()
  }, bg = "#050a15")

  output$event_dt <- renderDT({
    req(rv$events)
    rv$events %>%
      select(event_type, any_of(c("from_tp","to_tp","from_region",
                                   "to_region","region","score"))) %>%
      datatable(options = list(pageLength = 10, dom = "ftp"),
                rownames = FALSE, class = "stripe")
  })

  # ── Progression ──

  output$trajectory_plot <- renderPlot({
    req(rv$results, rv$region_data)
    rt <- rv$results$region_table
    if (nrow(rt) == 0) return(NULL)

    p1 <- ggplot(rt, aes(timepoint, volume_cm3,
                          color = region_type, group = region_type)) +
      geom_line(linewidth = 1.5) + geom_point(size = 3) +
      scale_color_manual(values = REGION_PAL, name = "Region") +
      labs(title = "Region Trajectories", x = "Timepoint",
           y = "cm\u00B3") + dark_theme()

    total <- rt %>% group_by(timepoint) %>%
      summarise(v = sum(volume_cm3, na.rm = TRUE), .groups = "drop")
    p2 <- ggplot(total, aes(timepoint, v, group = 1)) +
      geom_line(linewidth = 2, color = "#0088ff") +
      geom_point(size = 4, color = "#0088ff") +
      geom_text(aes(label = round(v, 1)), vjust = -1,
                color = "#fff", size = 3.5) +
      labs(title = "Total Volume", x = "Timepoint",
           y = "cm\u00B3") + dark_theme()

    p1 + p2 + plot_layout(widths = c(1.2, 1))
  }, bg = "#050a15")

  output$progression_card <- renderUI({
    prog <- rv$progression
    if (is.null(prog)) return(h4("Run analysis first", style="color:#3a5a7a;"))
    cls <- prog$classification
    conf <- round(prog$confidence * 100)
    clr <- switch(cls, CR="#22c55e", PR="#60a5fa", SD="#f59e0b",
                  PD="#ef4444", "#5a7a9a")
    lbl <- switch(cls,
      CR = "Complete Response", PR = "Partial Response",
      SD = "Stable Disease", PD = "Progressive Disease", cls)
    ev <- prog$evidence
    HTML(paste0(
      "<h4>RANO Classification</h4>",
      "<div style='font-size:42px;font-weight:900;color:", clr,
      ";text-shadow:0 0 20px ", clr, ";'>", cls, "</div>",
      "<div style='color:#c8ddf0;font-size:16px;'>", lbl, "</div>",
      "<div style='color:#5a7a9a;margin-top:4px;'>Confidence: ",
      conf, "%</div>",
      "<hr style='border-color:rgba(0,212,255,.1);'>",
      "<div style='color:#8aaabe;font-size:13px;'>Evidence: ",
      ev$n_growing, " growing \u00B7 ",
      ev$n_disappeared, " disappeared</div>"))
  })

  output$traj_dt <- renderDT({
    req(rv$trajectories)
    df <- map_dfr(names(rv$trajectories), function(rid) {
      t <- rv$trajectories[[rid]]
      tibble(Region = rid, Trend = t$trend,
             `First Vol` = round(t$first_volume, 1),
             `Last Vol` = round(t$last_volume, 1),
             Change = if (!is.na(t$percent_change))
               sprintf("%+.0f%%", t$percent_change * 100) else "N/A")
    })
    datatable(df, options = list(pageLength = 10, dom = "ftp",
      columnDefs = list(list(className = "dt-center", targets = "_all"))),
      rownames = FALSE, class = "stripe")
  })

  # ── Report ──

  output$full_report <- renderText({
    req(rv$results, rv$progression)
    pid <- rv$results$pid
    rt  <- rv$results$region_table
    prog <- rv$progression
    ev   <- prog$evidence
    lines <- c(
      strrep("=", 55),
      paste("PATIENT REPORT:", pid), strrep("=", 55),
      paste("Date:", Sys.Date()),
      "Model: 3D U-Net v2 (22.6M params) + Rule-based RANO", "",
      paste("CLASSIFICATION:", prog$classification),
      paste("CONFIDENCE:", round(prog$confidence*100), "%"), "",
      "REGION SUMMARY:",
      paste("  Total regions:", nrow(rt)),
      paste("  Timepoints:", paste(unique(rt$timepoint), collapse=", ")))
    if (!is.null(rv$trajectories)) {
      lines <- c(lines, "", "TRAJECTORIES:")
      for (rid in names(rv$trajectories)) {
        t <- rv$trajectories[[rid]]
        lines <- c(lines, sprintf("  %s: %s | %.1f -> %.1f cm3 | %+.0f%%",
          rid, t$trend, t$first_volume, t$last_volume,
          if (!is.na(t$percent_change)) t$percent_change*100 else 0))
      }
    }
    if (!is.null(rv$events) && nrow(rv$events) > 0) {
      lines <- c(lines, "", "EVENTS:")
      for (i in seq_len(min(nrow(rv$events), 20))) {
        e <- rv$events[i, ]
        lines <- c(lines, sprintf("  [%s] %s -> %s", e$event_type,
          if ("from_tp" %in% names(e) && !is.na(e$from_tp)) e$from_tp else "",
          if ("to_tp" %in% names(e) && !is.na(e$to_tp)) e$to_tp else ""))
      }
    }
    lines <- c(lines, "", strrep("=", 55))
    paste(lines, collapse = "\n")
  })

  # ── Pipeline Results ──

  output$fig_display <- renderImage({
    req(input$fig_sel)
    fp <- file.path(FIGURES_PATH, input$fig_sel)
    list(src = fp, contentType = "image/png", width = "100%",
         alt = input$fig_sel)
  }, deleteFile = FALSE)
}

shinyApp(ui, server)
