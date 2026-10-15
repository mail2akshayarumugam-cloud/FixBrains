# ──────────────────────────────────────────────────────────────
# Stage 7: Progression Analysis Engine
# Classify region trajectories and patient-level progression
# ──────────────────────────────────────────────────────────────

library(tidyverse)

if (!exists("UTILS_LOADED", envir = .GlobalEnv)) {
  SCRIPT_DIR <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) "R")
  source(file.path(SCRIPT_DIR, "utils.R"))
}

classify_region_trend <- function(volumes, threshold_growth = 0.25,
                                   threshold_shrink = -0.25) {
  volumes <- volumes[!is.na(volumes)]
  if (length(volumes) < 2) return("INSUFFICIENT_DATA")

  first <- volumes[1]
  last <- volumes[length(volumes)]

  if (first == 0 && last > 0) return("APPEARED")
  if (first > 0 && last == 0) return("DISAPPEARED")
  if (first == 0 && last == 0) return("ABSENT")

  pct_change <- (last - first) / first

  if (pct_change > threshold_growth) {
    return("GROWTH")
  } else if (pct_change < threshold_shrink) {
    return("SHRINKAGE")
  } else {
    return("STABLE")
  }
}

compute_growth_rate <- function(volumes, time_weeks) {
  volumes <- volumes[!is.na(volumes)]
  time_weeks <- time_weeks[!is.na(volumes)]
  if (length(volumes) < 2) return(NA_real_)

  dt <- diff(range(time_weeks))
  if (dt == 0) return(NA_real_)
  (last(volumes) - first(volumes)) / dt
}

compute_acceleration <- function(volumes, time_weeks) {
  if (length(volumes) < 3) return(NA_real_)

  rates <- diff(volumes) / diff(time_weeks)
  if (length(rates) < 2) return(NA_real_)

  time_mids <- time_weeks[-1] - diff(time_weeks) / 2
  dt <- diff(time_mids)
  if (any(dt == 0)) return(NA_real_)

  mean(diff(rates) / dt, na.rm = TRUE)
}

analyze_region_trajectory <- function(region_volumes, time_weeks) {
  trend <- classify_region_trend(region_volumes)
  growth_rate <- compute_growth_rate(region_volumes, time_weeks)
  accel <- compute_acceleration(region_volumes, time_weeks)

  first_vol <- region_volumes[1]
  last_vol <- region_volumes[length(region_volumes)]
  abs_change <- last_vol - first_vol
  pct_change <- if (first_vol > 0) abs_change / first_vol else NA

  list(
    trend = trend,
    growth_rate_cm3_per_week = growth_rate,
    acceleration = accel,
    absolute_change = abs_change,
    percent_change = pct_change,
    first_volume = first_vol,
    last_volume = last_vol,
    is_persistent_growth = trend == "GROWTH" && !is.na(growth_rate) && growth_rate > 0,
    n_observations = length(region_volumes)
  )
}

classify_patient_progression <- function(region_trajectories, events,
                                          threshold_pd = 0.25) {
  if (length(region_trajectories) == 0) {
    return(list(classification = "SD", confidence = 0.1,
                evidence = list(n_growing = 0, n_shrinking = 0, n_stable = 0,
                                n_appeared = 0, n_disappeared = 0,
                                has_splits = FALSE, has_new_regions = FALSE)))
  }

  trends <- sapply(region_trajectories, `[[`, "trend")
  growth_rates <- sapply(region_trajectories, `[[`, "growth_rate_cm3_per_week")

  evt_types <- if (!is.null(events) && nrow(events) > 0 && "event_type" %in% names(events)) {
    events$event_type
  } else character(0)

  has_significant_growth <- any(trends == "GROWTH")
  has_new_regions <- any(evt_types == "APPEARANCE")
  has_splits <- any(evt_types == "SPLIT")
  all_shrinking <- all(trends %in% c("SHRINKAGE", "DISAPPEARED", "ABSENT"))
  all_stable <- all(trends == "STABLE")
  all_gone <- all(trends %in% c("DISAPPEARED", "ABSENT"))

  if (all_gone) {
    progression <- "CR"
    confidence <- 0.9
  } else if (all_shrinking) {
    progression <- "PR"
    confidence <- 0.8
  } else if (all_stable && !has_new_regions) {
    progression <- "SD"
    confidence <- 0.7
  } else if (has_significant_growth || has_new_regions || has_splits) {
    progression <- "PD"
    confidence <- 0.85
  } else {
    max_pct <- max(sapply(region_trajectories, function(t) {
      abs(t$percent_change %||% 0)
    }), na.rm = TRUE)

    if (max_pct > threshold_pd) {
      any_positive <- any(sapply(region_trajectories, function(t) {
        (t$percent_change %||% 0) > threshold_pd
      }))
      progression <- if (any_positive) "PD" else "PR"
    } else {
      progression <- "SD"
    }
    confidence <- 0.5
  }

  list(
    classification = progression,
    confidence = confidence,
    evidence = list(
      n_growing = sum(trends == "GROWTH"),
      n_shrinking = sum(trends == "SHRINKAGE"),
      n_stable = sum(trends == "STABLE"),
      n_appeared = sum(trends == "APPEARED"),
      n_disappeared = sum(trends == "DISAPPEARED"),
      has_splits = has_splits,
      has_new_regions = has_new_regions
    )
  )
}

generate_patient_report <- function(patient_id, region_table, event_table,
                                     trajectories, progression) {
  cat("\n")
  cat(strrep("=", 60), "\n")
  cat("Patient: ", patient_id, "\n")
  cat(strrep("=", 60), "\n\n")

  for (i in seq_along(trajectories)) {
    t <- trajectories[[i]]
    cat(sprintf("Region %s:\n", names(trajectories)[i]))
    cat(sprintf("  Trend:       %s\n", t$trend))
    cat(sprintf("  Volume:      %.1f → %.1f cm³\n", t$first_volume, t$last_volume))
    cat(sprintf("  Change:      %+.1f%%\n", (t$percent_change %||% 0) * 100))
    if (!is.na(t$growth_rate_cm3_per_week)) {
      cat(sprintf("  Growth rate: %+.3f cm³/week\n", t$growth_rate_cm3_per_week))
    }
    cat("\n")
  }

  if (nrow(event_table) > 0) {
    cat("Events:\n")
    for (i in seq_len(nrow(event_table))) {
      e <- event_table[i, ]
      detail <- if ("region" %in% names(e) && !is.na(e$region)) e$region
                else if ("from_region" %in% names(e) && !is.na(e$from_region))
                  paste(e$from_region, "→", e$to_region)
                else ""
      cat(sprintf("  [%s] %s\n", e$event_type, detail))
    }
    cat("\n")
  }

  cat(sprintf("Framework Prediction: %s (confidence: %.0f%%)\n",
              progression$classification, progression$confidence * 100))
  cat(strrep("=", 60), "\n")

  invisible(NULL)
}

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0) return(y)
  if (is.na(x) || is.nan(x) || is.infinite(x)) return(y)
  x
}
