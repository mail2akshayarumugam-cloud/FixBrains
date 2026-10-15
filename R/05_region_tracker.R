# ──────────────────────────────────────────────────────────────
# Stage 5: Longitudinal Region Tracking
# Match regions across consecutive timepoints
# Detect: continuation, split, merge, appearance, disappearance
# ──────────────────────────────────────────────────────────────

library(tidyverse)

if (!exists("UTILS_LOADED", envir = .GlobalEnv)) {
  SCRIPT_DIR <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) "R")
  source(file.path(SCRIPT_DIR, "utils.R"))
}

compute_matching_score <- function(region_a, region_b,
                                    w_iou = 0.5, w_spatial = 0.3, w_volume = 0.2) {
  iou <- compute_iou(region_a$mask, region_b$mask)

  dist <- euclidean_distance(region_a$centroid, region_b$centroid)
  spatial_sim <- if (is.na(dist)) 0 else exp(-dist / 20)

  vol_a <- region_a$volume_cm3
  vol_b <- region_b$volume_cm3
  vol_sim <- 1 - abs(vol_a - vol_b) / max(vol_a, vol_b, 1e-6)
  vol_sim <- max(vol_sim, 0)

  type_bonus <- if (region_a$region_type == region_b$region_type) 0.1 else 0

  score <- w_iou * iou + w_spatial * spatial_sim + w_volume * vol_sim + type_bonus
  min(score, 1.0)
}

build_cost_matrix <- function(regions_t1, regions_t2) {
  n1 <- length(regions_t1)
  n2 <- length(regions_t2)
  if (n1 == 0 || n2 == 0) return(matrix(0, nrow = n1, ncol = n2))

  cost_mat <- matrix(0, nrow = n1, ncol = n2)
  for (i in seq_len(n1)) {
    for (j in seq_len(n2)) {
      cost_mat[i, j] <- compute_matching_score(regions_t1[[i]], regions_t2[[j]])
    }
  }

  rownames(cost_mat) <- sapply(regions_t1, `[[`, "region_id")
  colnames(cost_mat) <- sapply(regions_t2, `[[`, "region_id")
  cost_mat
}

match_regions_greedy <- function(cost_matrix, threshold = 0.15) {
  if (nrow(cost_matrix) == 0 || ncol(cost_matrix) == 0) {
    return(list(matches = list(), unmatched_t1 = seq_len(nrow(cost_matrix)),
                unmatched_t2 = seq_len(ncol(cost_matrix))))
  }

  matches <- list()
  matched_t1 <- c()
  matched_t2 <- c()

  flat_order <- order(cost_matrix, decreasing = TRUE)

  for (idx in flat_order) {
    i <- ((idx - 1) %% nrow(cost_matrix)) + 1
    j <- ((idx - 1) %/% nrow(cost_matrix)) + 1

    if (cost_matrix[i, j] < threshold) break
    if (i %in% matched_t1 || j %in% matched_t2) next

    matches <- c(matches, list(list(
      t1_idx = i,
      t2_idx = j,
      t1_id = rownames(cost_matrix)[i],
      t2_id = colnames(cost_matrix)[j],
      score = cost_matrix[i, j]
    )))
    matched_t1 <- c(matched_t1, i)
    matched_t2 <- c(matched_t2, j)
  }

  list(
    matches = matches,
    unmatched_t1 = setdiff(seq_len(nrow(cost_matrix)), matched_t1),
    unmatched_t2 = setdiff(seq_len(ncol(cost_matrix)), matched_t2)
  )
}

detect_splits <- function(cost_matrix, match_result, threshold = 0.10) {
  splits <- list()
  for (i in match_result$unmatched_t1) {
    scores <- cost_matrix[i, ]
    candidates <- which(scores >= threshold)
    matched_t2 <- if (length(match_result$matches) > 0) sapply(match_result$matches, `[[`, "t2_idx") else integer(0)
    candidates <- setdiff(candidates, matched_t2)

    if (length(candidates) >= 2) {
      splits <- c(splits, list(list(
        type = "SPLIT",
        source = rownames(cost_matrix)[i],
        targets = colnames(cost_matrix)[candidates],
        scores = scores[candidates]
      )))
    }
  }
  splits
}

detect_merges <- function(cost_matrix, match_result, threshold = 0.10) {
  merges <- list()
  for (j in match_result$unmatched_t2) {
    scores <- cost_matrix[, j]
    candidates <- which(scores >= threshold)
    matched_t1 <- if (length(match_result$matches) > 0) sapply(match_result$matches, `[[`, "t1_idx") else integer(0)
    candidates <- setdiff(candidates, matched_t1)

    if (length(candidates) >= 2) {
      merges <- c(merges, list(list(
        type = "MERGE",
        sources = rownames(cost_matrix)[candidates],
        target = colnames(cost_matrix)[j],
        scores = scores[candidates]
      )))
    }
  }
  merges
}

track_regions_pairwise <- function(regions_t1, regions_t2, tp1_name, tp2_name,
                                    match_threshold = 0.15) {
  cost_matrix <- build_cost_matrix(regions_t1, regions_t2)
  result <- match_regions_greedy(cost_matrix, match_threshold)

  splits <- detect_splits(cost_matrix, result)
  merges <- detect_merges(cost_matrix, result)

  events <- list()

  for (m in result$matches) {
    events <- c(events, list(list(
      type = "CONTINUATION",
      from_tp = tp1_name, to_tp = tp2_name,
      from_region = m$t1_id, to_region = m$t2_id,
      score = m$score
    )))
  }

  split_t1_ids <- as.character(unlist(lapply(splits, `[[`, "source")))
  merge_t2_ids <- as.character(unlist(lapply(merges, `[[`, "target")))

  for (i in result$unmatched_t1) {
    rid <- rownames(cost_matrix)[i]
    if (length(split_t1_ids) > 0 && rid %in% split_t1_ids) next
    events <- c(events, list(list(
      type = "DISAPPEARANCE",
      from_tp = tp1_name, to_tp = tp2_name,
      region = rid
    )))
  }

  for (j in result$unmatched_t2) {
    rid <- colnames(cost_matrix)[j]
    if (length(merge_t2_ids) > 0 && rid %in% merge_t2_ids) next
    events <- c(events, list(list(
      type = "APPEARANCE",
      from_tp = tp1_name, to_tp = tp2_name,
      region = rid
    )))
  }

  for (s in splits) events <- c(events, list(s))
  for (m in merges) events <- c(events, list(m))

  log_msg("  ", tp1_name, " → ", tp2_name, ": ",
          length(result$matches), " continuations, ",
          length(result$unmatched_t1) - length(splits), " disappearances, ",
          length(result$unmatched_t2) - length(merges), " appearances, ",
          length(splits), " splits, ",
          length(merges), " merges")

  list(
    cost_matrix = cost_matrix,
    match_result = result,
    events = events
  )
}

track_patient_regions <- function(regions_by_timepoint, match_threshold = 0.15) {
  tp_names <- names(regions_by_timepoint)
  if (length(tp_names) < 2) {
    log_msg("Need at least 2 timepoints for tracking", level = "WARN")
    return(list(tracking = list(), all_events = list()))
  }

  tracking <- list()
  all_events <- list()

  for (i in seq_len(length(tp_names) - 1)) {
    tp1 <- tp_names[i]
    tp2 <- tp_names[i + 1]

    result <- track_regions_pairwise(
      regions_by_timepoint[[tp1]],
      regions_by_timepoint[[tp2]],
      tp1, tp2, match_threshold
    )

    pair_name <- paste0(tp1, "_to_", tp2)
    tracking[[pair_name]] <- result
    all_events <- c(all_events, result$events)
  }

  list(tracking = tracking, all_events = all_events)
}

create_event_table <- function(all_events, patient_id) {
  if (length(all_events) == 0) {
    return(tibble(patient_id = character(0), event_type = character(0)))
  }

  map_dfr(all_events, function(e) {
    base <- tibble(
      patient_id = patient_id,
      event_type = e$type
    )

    if (e$type == "CONTINUATION") {
      base$from_tp <- e$from_tp
      base$to_tp <- e$to_tp
      base$from_region <- e$from_region
      base$to_region <- e$to_region
      base$score <- e$score
    } else if (e$type %in% c("APPEARANCE", "DISAPPEARANCE")) {
      base$from_tp <- e$from_tp
      base$to_tp <- e$to_tp
      base$region <- e$region
    } else if (e$type == "SPLIT") {
      base$source <- e$source
      base$targets <- paste(e$targets, collapse = ",")
    } else if (e$type == "MERGE") {
      base$sources <- paste(e$sources, collapse = ",")
      base$target <- e$target
    }

    base
  })
}
