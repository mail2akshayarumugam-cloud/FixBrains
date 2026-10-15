library(shiny)
library(tidyverse)
library(RNifti)
library(torch)
library(ggplot2)
library(patchwork)
library(viridis)

setwd(normalizePath(".", winslash = "/"))

source("R/utils.R")
source("R/01_data_loader.R")
source("R/02_preprocessing.R")
source("R/03_model_3d_unet.R")
source("R/04_region_extractor.R")
source("R/05_region_tracker.R")
source("R/06_feature_extraction.R")
source("R/07_progression_engine.R")
source("R/11_inference.R")

IMAGING_ROOT <- "data/raw/LUMIERE/Imaging"
MODEL_PATH <- "models/unet_3d_trained_jit.pt"
TARGET_SIZE <- c(48L, 48L, 48L)

patient_list <- sort(list.dirs(IMAGING_ROOT, recursive = FALSE, full.names = FALSE))
patient_list <- patient_list[grepl("^Patient-", patient_list)]

message("Loading model...")
UNET_MODEL <- tryCatch({
  m <- jit_load(MODEL_PATH)
  m$to(device = "cpu")
  m
}, error = function(e) { message("Model not found"); NULL })

SEG_PAL <- c("0"="#111","1"="#ef4444","2"="#f59e0b","3"="#22c55e","4"="#60a5fa")

ui <- fluidPage(
  tags$head(tags$style(HTML("
    body { background:#0a0a1a; color:#ddd; font-family:Segoe UI,sans-serif; }
    .well { background:#151530; border:1px solid #2a2a4a; }
    .btn-primary { background:#0066cc; border:none; padding:12px 20px; font-size:15px; font-weight:700; }
    .btn-info { background:#224; border:1px solid #0088ff; padding:10px 20px; }
    .nav-tabs>li>a { color:#888; }
    .nav-tabs>li.active>a { color:#fff; background:#1a1a3a; border-color:#0088ff; }
    .tab-content { background:#0f0f2a; border:1px solid #2a2a4a; padding:20px; border-radius:0 0 8px 8px; }
    h3 { color:#0088ff; }
    pre { background:#0a0a1a; color:#aaa; border:1px solid #2a2a4a; }
    table { color:#ddd !important; }
  "))),

  titlePanel(div(style="color:#0088ff;", "Tumor Progression Analysis")),

  sidebarLayout(
    sidebarPanel(width = 3,
      selectInput("patient_id", "Select Patient:", patient_list, selected = "Patient-083"),
      textInput("custom_folder", "Or paste folder path:", placeholder = "C:/path/to/Patient-XXX"),
      hr(),
      actionButton("preview_btn", "Preview MRI", class = "btn-info", style = "width:100%; margin-bottom:10px;"),
      actionButton("run_btn", "Run Full Analysis", class = "btn-primary", style = "width:100%;"),
      hr(),
      verbatimTextOutput("status")
    ),

    mainPanel(width = 9,
      tabsetPanel(id = "tabs",
        tabPanel("MRI Preview",
          br(),
          fluidRow(
            column(6, selectInput("pv_tp", "Timepoint:", choices = NULL)),
            column(6, sliderInput("pv_slice", "Slice:", 1, 48, 24))
          ),
          plotOutput("mri_preview_plot", height = "400px")
        ),
        tabPanel("Segmentation",
          br(),
          plotOutput("seg_plot", height = "400px")
        ),
        tabPanel("Regions",
          br(),
          plotOutput("region_plot", height = "350px"),
          br(),
          tableOutput("region_table")
        ),
        tabPanel("Progression",
          br(),
          htmlOutput("progression_result"),
          br(),
          plotOutput("trajectory_plot", height = "350px")
        ),
        tabPanel("Report",
          br(),
          verbatimTextOutput("report_text")
        )
      )
    )
  )
)

server <- function(input, output, session) {
  rv <- reactiveValues(
    status = "Ready",
    preview = NULL,
    patient_data = NULL,
    preprocessed = NULL,
    segmented = NULL,
    region_data = NULL,
    events = NULL,
    trajectories = NULL,
    progression = NULL
  )

  output$status <- renderText({ rv$status })

  get_path <- function() {
    if (nzchar(trimws(input$custom_folder))) {
      normalizePath(trimws(input$custom_folder), winslash = "/", mustWork = FALSE)
    } else {
      file.path(IMAGING_ROOT, input$patient_id)
    }
  }

  get_pid <- function() {
    if (nzchar(trimws(input$custom_folder))) {
      basename(trimws(input$custom_folder))
    } else {
      input$patient_id
    }
  }

  # ── PREVIEW ──
  observeEvent(input$preview_btn, {
    pp <- get_path()
    rv$status <- paste("Loading preview from:", pp)

    if (!dir.exists(pp)) {
      rv$status <- paste("Folder not found:", pp)
      return()
    }

    pd <- tryCatch(load_full_patient(pp), error = function(e) {
      rv$status <- paste("Error:", e$message)
      NULL
    })
    if (is.null(pd)) return()

    rv$status <- "Preprocessing..."
    preprocessed <- tryCatch(
      preprocess_full_patient(pd, target_size = TARGET_SIZE),
      error = function(e) { rv$status <- paste("Error:", e$message); NULL }
    )
    if (is.null(preprocessed)) return()

    rv$patient_data <- pd
    rv$preprocessed <- preprocessed

    prev <- list()
    mod_names <- c("T1", "T1c", "T2", "FLAIR")
    for (tp_name in names(preprocessed)) {
      tp <- preprocessed[[tp_name]]
      if (is.null(tp$stacked)) next
      vols <- list()
      for (ch in seq_len(min(dim(tp$stacked)[4], 4))) {
        vols[[mod_names[ch]]] <- tp$stacked[, , , ch]
      }
      prev[[tp_name]] <- list(volumes = vols, seg = tp$segmentation, dims = dim(tp$stacked)[1:3])
    }

    rv$preview <- prev
    tps <- names(prev)
    updateSelectInput(session, "pv_tp", choices = tps, selected = tps[1])
    if (length(prev) > 0) {
      updateSliderInput(session, "pv_slice", min = 1, max = prev[[1]]$dims[3], value = prev[[1]]$dims[3] %/% 2)
    }
    updateTabsetPanel(session, "tabs", selected = "MRI Preview")
    rv$status <- paste("Preview loaded:", length(prev), "timepoints")
    rm(pd); gc()
  })

  # ── MRI PREVIEW PLOT ──
  output$mri_preview_plot <- renderPlot({
    req(rv$preview, input$pv_tp)
    tp <- rv$preview[[input$pv_tp]]
    if (is.null(tp)) return(NULL)
    z <- max(1, min(input$pv_slice, tp$dims[3]))

    plots <- list()
    for (m in names(tp$volumes)) {
      v <- tp$volumes[[m]]
      sl <- v[, , z]
      df <- expand.grid(x = seq_len(nrow(sl)), y = seq_len(ncol(sl)))
      df$v <- as.vector(sl)
      plots[[m]] <- ggplot(df, aes(x, y, fill = v)) +
        geom_raster() +
        scale_fill_viridis_c(option = "inferno", guide = "none") +
        coord_fixed() + labs(title = m) +
        theme_void() +
        theme(plot.title = element_text(hjust = .5, color = "white", face = "bold", size = 12),
              plot.background = element_rect(fill = "#0a0a1a", color = NA))
    }

    if (!is.null(tp$seg)) {
      seg <- tp$seg
      sl <- seg[, , z]
      df <- expand.grid(x = seq_len(nrow(sl)), y = seq_len(ncol(sl)))
      df$label <- factor(as.vector(sl))
      plots[["Segmentation"]] <- ggplot(df, aes(x, y, fill = label)) +
        geom_raster() +
        scale_fill_manual(values = SEG_PAL, guide = "none") +
        coord_fixed() + labs(title = "Segmentation") +
        theme_void() +
        theme(plot.title = element_text(hjust = .5, color = "white", face = "bold", size = 12),
              plot.background = element_rect(fill = "#0a0a1a", color = NA))
    }

    if (length(plots) == 0) return(NULL)
    wrap_plots(plots, nrow = 1) +
      plot_annotation(theme = theme(plot.background = element_rect(fill = "#0a0a1a", color = NA)))
  }, bg = "#0a0a1a")

  # ── RUN ANALYSIS ──
  observeEvent(input$run_btn, {
    pp <- get_path()
    pid <- get_pid()
    rv$status <- paste("Analyzing", pid, "...")

    if (!dir.exists(pp)) { rv$status <- "Folder not found"; return() }
    if (is.null(UNET_MODEL)) { rv$status <- "Model not loaded"; return() }

    withProgress(message = paste("Analyzing", pid), value = 0, {

      if (is.null(rv$preprocessed)) {
        incProgress(0.1, detail = "Loading...")
        pd <- tryCatch(load_full_patient(pp), error = function(e) NULL)
        if (is.null(pd)) { rv$status <- "Load failed"; return() }
        rv$patient_data <- pd

        incProgress(0.2, detail = "Preprocessing...")
        preprocessed <- tryCatch(
          preprocess_full_patient(pd, target_size = TARGET_SIZE),
          error = function(e) NULL
        )
        if (is.null(preprocessed)) { rv$status <- "Preprocess failed"; return() }
        rv$preprocessed <- preprocessed
      } else {
        incProgress(0.3, detail = "Using cached data...")
        preprocessed <- rv$preprocessed
      }

      incProgress(0.2, detail = "U-Net segmentation...")
      rv$status <- "Running 3D U-Net..."
      segmented <- segment_patient(UNET_MODEL, preprocessed, "cpu")
      rv$segmented <- segmented

      incProgress(0.15, detail = "Extracting regions...")
      rd <- extract_all_patient_regions(segmented, pid, min_vol = 50)
      rv$region_data <- rd

      incProgress(0.1, detail = "Tracking...")
      events <- NULL
      trajectories <- NULL
      progression <- NULL

      if (length(rd$regions_by_timepoint) >= 2) {
        tracking <- track_patient_regions(rd$regions_by_timepoint)
        events <- create_event_table(tracking$all_events, pid)

        tp_names <- names(rd$regions_by_timepoint)
        pd <- rv$patient_data
        time_weeks <- sapply(pd$timepoints[tp_names], function(x) x$week_number)

        trajectories <- list()
        for (r in rd$regions_by_timepoint[[1]]) {
          vols <- sapply(tp_names, function(tp) {
            matched <- Filter(function(x) x$region_type == r$region_type,
                              rd$regions_by_timepoint[[tp]])
            if (length(matched) > 0) matched[[1]]$volume_cm3 else 0
          })
          trajectories[[r$region_id]] <- analyze_region_trajectory(vols, time_weeks)
        }

        if (is.null(events)) events <- tibble(event_type = character(0))
        progression <- classify_patient_progression(trajectories, events)
      }

      rv$events <- events
      rv$trajectories <- trajectories
      rv$progression <- progression

      incProgress(0.15, detail = "Done!")
      rv$status <- paste("Done!", pid, "-",
        if (!is.null(progression)) progression$classification else "N/A")
    })
  })

  # ── SEGMENTATION PLOT ──
  output$seg_plot <- renderPlot({
    req(rv$segmented)
    seg_data <- rv$segmented
    plots <- list()
    for (tp in names(seg_data)) {
      seg <- seg_data[[tp]]$segmentation
      if (is.null(seg)) next
      z <- dim(seg)[3] %/% 2
      sl <- seg[, , z]
      df <- expand.grid(x = seq_len(nrow(sl)), y = seq_len(ncol(sl)))
      df$label <- factor(as.vector(sl))
      plots[[tp]] <- ggplot(df, aes(x, y, fill = label)) +
        geom_raster() +
        scale_fill_manual(values = SEG_PAL, guide = "none") +
        coord_fixed() + labs(title = tp) +
        theme_void() +
        theme(plot.title = element_text(hjust = .5, color = "white", face = "bold"),
              plot.background = element_rect(fill = "#0a0a1a", color = NA))
    }
    if (length(plots) == 0) return(NULL)
    wrap_plots(plots, nrow = 1) +
      plot_annotation(title = "U-Net Segmentation per Timepoint",
        theme = theme(plot.title = element_text(color = "white", face = "bold", size = 16),
                      plot.background = element_rect(fill = "#0a0a1a", color = NA)))
  }, bg = "#0a0a1a")

  # ── REGION PLOT ──
  output$region_plot <- renderPlot({
    req(rv$region_data)
    rt <- rv$region_data$table
    if (nrow(rt) == 0) return(NULL)
    ggplot(rt, aes(x = timepoint, y = volume_cm3, fill = region_type)) +
      geom_col(position = "dodge", alpha = 0.85) +
      scale_fill_manual(values = c(necrosis="#ef4444", edema="#f59e0b",
                                   enhancing="#22c55e", label_4="#60a5fa")) +
      labs(title = "Region Volumes", x = "Timepoint", y = "Volume (cm³)", fill = "Region") +
      theme_minimal(base_size = 14) +
      theme(plot.background = element_rect(fill = "#0a0a1a", color = NA),
            panel.background = element_rect(fill = "#0a0a1a", color = NA),
            text = element_text(color = "#ddd"),
            axis.text = element_text(color = "#888"),
            panel.grid = element_line(color = "#222"),
            plot.title = element_text(face = "bold", color = "white"))
  }, bg = "#0a0a1a")

  output$region_table <- renderTable({
    req(rv$region_data)
    rv$region_data$table %>%
      select(timepoint, region_id, region_type, volume_cm3, n_voxels) %>%
      mutate(volume_cm3 = round(volume_cm3, 2))
  }, striped = TRUE)

  # ── PROGRESSION ──
  output$progression_result <- renderUI({
    prog <- rv$progression
    if (is.null(prog)) return(h3("Run analysis first", style = "color:#666;"))

    cls <- prog$classification
    conf <- round(prog$confidence * 100)
    color <- switch(cls, CR="#22c55e", PR="#60a5fa", SD="#f59e0b", PD="#ef4444", "#888")
    label <- switch(cls,
      CR = "Complete Response — tumor gone",
      PR = "Partial Response — tumor shrinking",
      SD = "Stable Disease — no significant change",
      PD = "Progressive Disease — tumor growing",
      cls)

    ev <- prog$evidence
    HTML(paste0(
      "<div style='background:#151530;border:2px solid ", color, ";border-radius:12px;padding:24px;text-align:center;'>",
      "<div style='font-size:48px;font-weight:900;color:", color, ";'>", cls, "</div>",
      "<div style='font-size:18px;color:#ccc;margin:8px 0;'>", label, "</div>",
      "<div style='font-size:14px;color:#888;'>Confidence: ", conf, "%</div>",
      "<hr style='border-color:#333;'>",
      "<div style='text-align:left;font-size:13px;color:#aaa;'>",
      "Growing: ", ev$n_growing, " | Shrinking: ", ev$n_shrinking,
      " | Stable: ", ev$n_stable, " | Appeared: ", ev$n_appeared,
      " | Disappeared: ", ev$n_disappeared,
      "</div></div>"
    ))
  })

  output$trajectory_plot <- renderPlot({
    req(rv$region_data)
    rt <- rv$region_data$table
    if (nrow(rt) == 0) return(NULL)

    total <- rt %>% group_by(timepoint) %>%
      summarise(total_vol = sum(volume_cm3, na.rm = TRUE), .groups = "drop")

    ggplot(total, aes(x = timepoint, y = total_vol, group = 1)) +
      geom_line(linewidth = 2, color = "#0088ff") +
      geom_point(size = 4, color = "#0088ff") +
      geom_text(aes(label = round(total_vol, 1)), vjust = -1, color = "white", size = 4) +
      labs(title = "Total Tumor Volume Over Time", x = "Timepoint", y = "Volume (cm³)") +
      theme_minimal(base_size = 14) +
      theme(plot.background = element_rect(fill = "#0a0a1a", color = NA),
            panel.background = element_rect(fill = "#0a0a1a", color = NA),
            text = element_text(color = "#ddd"),
            axis.text = element_text(color = "#888"),
            panel.grid = element_line(color = "#222"),
            plot.title = element_text(face = "bold", color = "white"))
  }, bg = "#0a0a1a")

  # ── REPORT ──
  output$report_text <- renderText({
    req(rv$progression)
    pid <- get_pid()
    prog <- rv$progression
    rt <- rv$region_data$table
    ev <- prog$evidence

    lines <- c(
      paste(strrep("=", 50)),
      paste("PATIENT:", pid),
      paste("CLASSIFICATION:", prog$classification, "(", round(prog$confidence*100), "% confidence)"),
      paste(strrep("=", 50)),
      "",
      paste("Regions found:", nrow(rt)),
      paste("Growing:", ev$n_growing),
      paste("Shrinking:", ev$n_shrinking),
      paste("Stable:", ev$n_stable),
      paste("Appeared:", ev$n_appeared),
      paste("Disappeared:", ev$n_disappeared),
      ""
    )

    if (!is.null(rv$trajectories)) {
      lines <- c(lines, "TRAJECTORIES:")
      for (rid in names(rv$trajectories)) {
        t <- rv$trajectories[[rid]]
        lines <- c(lines, sprintf("  %s: %s | %.1f -> %.1f cm3",
          rid, t$trend, t$first_volume, t$last_volume))
      }
    }

    paste(lines, collapse = "\n")
  })
}

shinyApp(ui, server)
