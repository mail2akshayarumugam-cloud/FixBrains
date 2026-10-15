# ──────────────────────────────────────────────────────────────
# Stage 9: Clinical Validation
# Compare framework predictions against RANO assessments
# ──────────────────────────────────────────────────────────────

library(tidyverse)
library(irr)
library(caret)
library(pROC)

if (!exists("UTILS_LOADED", envir = .GlobalEnv)) {
  SCRIPT_DIR <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) "R")
  source(file.path(SCRIPT_DIR, "utils.R"))
}

compute_confusion_matrix <- function(predicted, actual, classes = c("CR", "PR", "SD", "PD")) {
  predicted <- factor(predicted, levels = classes)
  actual <- factor(actual, levels = classes)
  table(Predicted = predicted, Actual = actual)
}

compute_classification_metrics <- function(predicted, actual,
                                            classes = c("CR", "PR", "SD", "PD")) {
  predicted <- factor(predicted, levels = classes)
  actual <- factor(actual, levels = classes)

  cm <- tryCatch(
    confusionMatrix(predicted, actual),
    error = function(e) {
      list(table = table(Predicted = predicted, Actual = actual))
    }
  )

  per_class <- list()
  for (cls in classes) {
    tp <- sum(predicted == cls & actual == cls)
    fp <- sum(predicted == cls & actual != cls)
    fn <- sum(predicted != cls & actual == cls)
    tn <- sum(predicted != cls & actual != cls)

    precision <- if ((tp + fp) > 0) tp / (tp + fp) else 0
    recall <- if ((tp + fn) > 0) tp / (tp + fn) else 0
    f1 <- if ((precision + recall) > 0) 2 * precision * recall / (precision + recall) else 0
    specificity <- if ((tn + fp) > 0) tn / (tn + fp) else 0

    per_class[[cls]] <- list(
      precision = precision,
      recall = recall,
      f1 = f1,
      specificity = specificity,
      support = tp + fn
    )
  }

  overall_accuracy <- sum(predicted == actual) / length(actual)

  macro_precision <- mean(sapply(per_class, `[[`, "precision"))
  macro_recall <- mean(sapply(per_class, `[[`, "recall"))
  macro_f1 <- mean(sapply(per_class, `[[`, "f1"))

  weights <- sapply(per_class, `[[`, "support")
  weights <- weights / sum(weights)
  weighted_f1 <- sum(sapply(per_class, `[[`, "f1") * weights)

  list(
    confusion_matrix = cm$table,
    overall_accuracy = overall_accuracy,
    macro_precision = macro_precision,
    macro_recall = macro_recall,
    macro_f1 = macro_f1,
    weighted_f1 = weighted_f1,
    per_class = per_class
  )
}

compute_cohens_kappa <- function(predicted, actual,
                                  classes = c("CR", "PR", "SD", "PD")) {
  predicted <- factor(predicted, levels = classes)
  actual <- factor(actual, levels = classes)

  ratings <- data.frame(rater1 = as.integer(predicted), rater2 = as.integer(actual))
  kappa_result <- kappa2(ratings, weight = "unweighted")

  list(
    kappa = kappa_result$value,
    z = kappa_result$statistic,
    p_value = kappa_result$p.value
  )
}

compute_binary_progression_metrics <- function(predicted, actual) {
  pred_binary <- ifelse(predicted == "PD", "Progression", "No Progression")
  actual_binary <- ifelse(actual == "PD", "Progression", "No Progression")

  pred_f <- factor(pred_binary, levels = c("Progression", "No Progression"))
  actual_f <- factor(actual_binary, levels = c("Progression", "No Progression"))

  tp <- sum(pred_f == "Progression" & actual_f == "Progression")
  fp <- sum(pred_f == "Progression" & actual_f == "No Progression")
  fn <- sum(pred_f == "No Progression" & actual_f == "Progression")
  tn <- sum(pred_f == "No Progression" & actual_f == "No Progression")

  list(
    sensitivity = if ((tp + fn) > 0) tp / (tp + fn) else 0,
    specificity = if ((tn + fp) > 0) tn / (tn + fp) else 0,
    ppv = if ((tp + fp) > 0) tp / (tp + fp) else 0,
    npv = if ((tn + fn) > 0) tn / (tn + fn) else 0,
    accuracy = (tp + tn) / (tp + fp + fn + tn),
    f1 = if ((tp + fp) > 0 && (tp + fn) > 0) {
      prec <- tp / (tp + fp)
      rec <- tp / (tp + fn)
      2 * prec * rec / (prec + rec)
    } else 0
  )
}

run_ablation_study <- function(patient_features, rano_labels) {
  results <- list()

  pf <- patient_features %>%
    mutate(across(where(is.numeric), ~ifelse(is.na(.) | is.infinite(.) | is.nan(.), 0, .)))

  # Experiment 1: Total volume only
  exp1_pred <- pf %>%
    mutate(prediction = case_when(
      total_vol_pct_change > 0.25 ~ "PD",
      total_vol_pct_change < -0.50 ~ "CR",
      total_vol_pct_change < -0.25 ~ "PR",
      TRUE ~ "SD"
    )) %>%
    left_join(rano_labels, by = "patient_id") %>% filter(!is.na(rano_label))

  results$total_volume <- list(
    name = "Total Volume Only",
    metrics = compute_classification_metrics(exp1_pred$prediction, exp1_pred$rano_label),
    kappa = compute_cohens_kappa(exp1_pred$prediction, exp1_pred$rano_label),
    binary = compute_binary_progression_metrics(exp1_pred$prediction, exp1_pred$rano_label)
  )

  # Experiment 2: Region-wise volume
  exp2_pred <- pf %>%
    mutate(prediction = case_when(
      n_growing_regions >= 2 | max_growth_rate > 0.5 ~ "PD",
      n_growing_regions == 1 & n_appearances > 0 ~ "PD",
      n_shrinking_regions > n_growing_regions & total_vol_pct_change < -0.50 ~ "CR",
      n_shrinking_regions > n_growing_regions ~ "PR",
      TRUE ~ "SD"
    )) %>%
    left_join(rano_labels, by = "patient_id") %>% filter(!is.na(rano_label))

  results$region_wise <- list(
    name = "Region-Wise Volume",
    metrics = compute_classification_metrics(exp2_pred$prediction, exp2_pred$rano_label),
    kappa = compute_cohens_kappa(exp2_pred$prediction, exp2_pred$rano_label),
    binary = compute_binary_progression_metrics(exp2_pred$prediction, exp2_pred$rano_label)
  )

  # Experiment 3: Full framework
  exp3_pred <- pf %>%
    mutate(prediction = case_when(
      n_growing_regions >= 2 | max_growth_rate > 0.4 | n_appearances >= 2 ~ "PD",
      n_growing_regions == 1 & (n_appearances > 0 | n_splits > 0) ~ "PD",
      max_growth_rate > 0.3 & max_centroid_displacement > 10 ~ "PD",
      n_shrinking_regions > n_growing_regions &
        total_vol_pct_change < -0.50 &
        n_disappearances > 0 ~ "CR",
      n_shrinking_regions > n_growing_regions &
        total_vol_pct_change < -0.25 ~ "PR",
      total_events == 0 & abs(total_vol_pct_change) < 0.10 ~ "SD",
      TRUE ~ "SD"
    )) %>%
    left_join(rano_labels, by = "patient_id") %>% filter(!is.na(rano_label))

  results$full_framework <- list(
    name = "Full Longitudinal Framework",
    metrics = compute_classification_metrics(exp3_pred$prediction, exp3_pred$rano_label),
    kappa = compute_cohens_kappa(exp3_pred$prediction, exp3_pred$rano_label),
    binary = compute_binary_progression_metrics(exp3_pred$prediction, exp3_pred$rano_label)
  )

  results
}

format_ablation_table <- function(ablation_results) {
  map_dfr(names(ablation_results), function(name) {
    r <- ablation_results[[name]]
    tibble(
      Approach = r$name,
      Accuracy = sprintf("%.1f%%", r$metrics$overall_accuracy * 100),
      `Macro F1` = sprintf("%.3f", r$metrics$macro_f1),
      `Weighted F1` = sprintf("%.3f", r$metrics$weighted_f1),
      Kappa = sprintf("%.3f", r$kappa$kappa),
      `PD Sensitivity` = sprintf("%.3f", r$binary$sensitivity),
      `PD Specificity` = sprintf("%.3f", r$binary$specificity),
      `PD F1` = sprintf("%.3f", r$binary$f1)
    )
  })
}

print_validation_report <- function(ablation_results) {
  cat("\n")
  cat(strrep("=", 80), "\n")
  cat("ABLATION STUDY: Clinical Validation Results\n")
  cat(strrep("=", 80), "\n\n")

  table <- format_ablation_table(ablation_results)
  print(table, n = Inf)

  cat("\n")
  cat(strrep("-", 80), "\n")

  for (name in names(ablation_results)) {
    r <- ablation_results[[name]]
    cat(sprintf("\n[%s] Per-class breakdown:\n", r$name))
    for (cls in names(r$metrics$per_class)) {
      pc <- r$metrics$per_class[[cls]]
      cat(sprintf("  %s: P=%.3f  R=%.3f  F1=%.3f  (n=%d)\n",
                  cls, pc$precision, pc$recall, pc$f1, pc$support))
    }
  }

  cat("\n")
  cat(strrep("=", 80), "\n")
}
