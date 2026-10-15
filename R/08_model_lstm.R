# ──────────────────────────────────────────────────────────────
# Stage 8: LSTM for Temporal Progression Prediction (R torch)
# Takes region feature sequences → predicts RANO class
# ──────────────────────────────────────────────────────────────

library(torch)
library(luz)

if (!exists("UTILS_LOADED", envir = .GlobalEnv)) {
  SCRIPT_DIR <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) "R")
  source(file.path(SCRIPT_DIR, "utils.R"))
}

# ── LSTM Model ────────────────────────────────────────────────

progression_lstm <- nn_module(
  "ProgressionLSTM",
  initialize = function(input_size, hidden_size = 64L, num_layers = 2L,
                         n_classes = 4L, dropout = 0.3) {
    self$lstm <- nn_lstm(
      input_size = input_size,
      hidden_size = hidden_size,
      num_layers = num_layers,
      batch_first = TRUE,
      dropout = if (num_layers > 1) dropout else 0
    )

    self$attention <- nn_linear(hidden_size, 1L)

    self$classifier <- nn_sequential(
      nn_linear(hidden_size, hidden_size %/% 2L),
      nn_relu(),
      nn_dropout(p = dropout),
      nn_linear(hidden_size %/% 2L, n_classes)
    )
  },
  forward = function(x, lengths = NULL) {
    lstm_out <- self$lstm(x)
    output <- lstm_out[[1]]

    attn_weights <- self$attention(output)$squeeze(-1)
    attn_weights <- nnf_softmax(attn_weights, dim = 2)

    context <- torch_bmm(
      attn_weights$unsqueeze(2),
      output
    )$squeeze(2)

    self$classifier(context)
  }
)

# ── Feature Preparation ──────────────────────────────────────

prepare_lstm_features <- function(patient_summary_features, region_table,
                                   event_table, long_features,
                                   rano_labels, max_seq_len = 20L) {
  patients <- unique(patient_summary_features$patient_id)

  feature_cols <- c(
    "total_vol_change", "total_vol_pct_change",
    "n_growing_regions", "n_shrinking_regions",
    "max_growth_rate", "max_centroid_displacement",
    "n_appearances", "n_disappearances",
    "n_splits", "n_merges", "total_events"
  )

  sequences <- list()
  labels <- c()

  for (pid in patients) {
    pt_regions <- region_table %>% filter(patient_id == pid)
    timepoints <- sort(unique(pt_regions$timepoint))

    if (length(timepoints) < 2) next

    rano <- rano_labels %>% filter(patient_id == pid)
    if (nrow(rano) == 0) next

    seq_features <- matrix(0, nrow = max_seq_len,
                           ncol = length(feature_cols) + 4)

    for (t_idx in seq_along(timepoints)) {
      if (t_idx > max_seq_len) break
      tp <- timepoints[t_idx]

      tp_regions <- pt_regions %>% filter(timepoint == tp)
      total_vol <- sum(tp_regions$volume_cm3, na.rm = TRUE)
      n_regions <- nrow(tp_regions)
      max_vol <- max(tp_regions$volume_cm3, na.rm = TRUE)
      mean_vol <- mean(tp_regions$volume_cm3, na.rm = TRUE)

      feat_vec <- c(total_vol, n_regions, max_vol, mean_vol)

      if (t_idx > 1) {
        prev_tp <- timepoints[t_idx - 1]
        prev_vol <- pt_regions %>%
          filter(timepoint == prev_tp) %>%
          summarise(v = sum(volume_cm3, na.rm = TRUE)) %>%
          pull(v)
        vol_change <- total_vol - prev_vol
        vol_pct <- if (prev_vol > 0) vol_change / prev_vol else 0

        tp_events <- event_table %>%
          filter(patient_id == pid, to_tp == tp)
        n_appear <- sum(tp_events$event_type == "APPEARANCE")
        n_disappear <- sum(tp_events$event_type == "DISAPPEARANCE")
        n_split <- sum(tp_events$event_type == "SPLIT")
        n_merge <- sum(tp_events$event_type == "MERGE")

        tp_long <- long_features %>% filter(patient_id == pid, to_tp == tp)
        max_growth <- if (nrow(tp_long) > 0) max(tp_long$pct_volume_change, na.rm = TRUE) else 0
        max_disp <- if (nrow(tp_long) > 0) max(tp_long$centroid_displacement, na.rm = TRUE) else 0
        n_growing <- if (nrow(tp_long) > 0) sum(tp_long$pct_volume_change > 0.25, na.rm = TRUE) else 0

        feat_vec <- c(feat_vec, vol_change, vol_pct, max_growth, max_disp,
                       n_growing, n_appear, n_disappear, n_split, n_merge)
      } else {
        feat_vec <- c(feat_vec, rep(0, 9))
      }

      n_feat <- min(length(feat_vec), ncol(seq_features))
      seq_features[t_idx, 1:n_feat] <- feat_vec[1:n_feat]
    }

    sequences <- c(sequences, list(seq_features))

    rano_class <- match(rano$rano_label[1], c("CR", "PR", "SD", "PD")) - 1L
    labels <- c(labels, rano_class)
  }

  list(
    sequences = sequences,
    labels = labels,
    feature_names = c("total_vol", "n_regions", "max_vol", "mean_vol",
                       "vol_change", "vol_pct", "max_growth", "max_disp",
                       "n_growing", "n_appear", "n_disappear", "n_split", "n_merge"),
    n_patients = length(sequences)
  )
}

# ── Dataset ───────────────────────────────────────────────────

progression_dataset <- dataset(
  "ProgressionDataset",
  initialize = function(sequences, labels) {
    self$sequences <- sequences
    self$labels <- labels
  },
  .getitem = function(i) {
    x <- torch_tensor(self$sequences[[i]], dtype = torch_float())
    y <- torch_tensor(as.integer(self$labels[[i]]), dtype = torch_long())$squeeze()
    list(x = x, y = y)
  },
  .length = function() {
    length(self$sequences)
  }
)

# ── Training ──────────────────────────────────────────────────

train_lstm <- function(train_sequences, train_labels,
                       val_sequences = NULL, val_labels = NULL,
                       input_size = 13L, hidden_size = 64L,
                       num_layers = 2L, n_classes = 4L,
                       epochs = 100L, batch_size = 16L, lr = 1e-3,
                       save_path = "models/lstm_progression.pt") {

  device <- if (cuda_is_available()) "cuda" else "cpu"
  log_msg("Training LSTM on: ", device)

  train_ds <- progression_dataset(train_sequences, train_labels)
  train_dl <- dataloader(train_ds, batch_size = batch_size, shuffle = TRUE)

  val_dl <- NULL
  if (!is.null(val_sequences)) {
    val_ds <- progression_dataset(val_sequences, val_labels)
    val_dl <- dataloader(val_ds, batch_size = batch_size, shuffle = FALSE)
  }

  model <- progression_lstm(input_size, hidden_size, num_layers, n_classes)
  model <- model$to(device = device)
  optimizer <- optim_adam(model$parameters, lr = lr, weight_decay = 1e-4)

  class_counts <- table(factor(train_labels, levels = 0:(n_classes - 1L)))
  class_counts[class_counts == 0] <- 1
  class_weights <- 1.0 / (class_counts / sum(class_counts))
  class_weights <- class_weights / sum(class_weights) * length(class_weights)
  weight_tensor <- torch_tensor(as.numeric(class_weights), dtype = torch_float())$to(device = device)

  best_val_loss <- Inf
  best_val_acc <- 0
  patience_counter <- 0
  patience <- 15

  history <- list(train_loss = c(), train_acc = c(), val_loss = c(), val_acc = c())

  for (epoch in seq_len(epochs)) {
    model$train()
    train_losses <- c()
    train_correct <- 0
    train_total <- 0

    coro::loop(for (batch in train_dl) {
      x <- batch$x$to(device = device)
      y <- batch$y$to(device = device)

      optimizer$zero_grad()
      pred <- model(x)
      loss <- nnf_cross_entropy(pred, y, weight = weight_tensor)
      loss$backward()
      nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
      optimizer$step()

      train_losses <- c(train_losses, loss$item())
      predicted <- pred$argmax(dim = 2)
      train_correct <- train_correct + (predicted == y)$sum()$item()
      train_total <- train_total + y$size(1)
    })

    avg_train_loss <- mean(train_losses)
    train_acc <- train_correct / train_total
    history$train_loss <- c(history$train_loss, avg_train_loss)
    history$train_acc <- c(history$train_acc, train_acc)

    msg <- sprintf("Epoch %d/%d — Loss: %.4f, Acc: %.2f%%",
                   epoch, epochs, avg_train_loss, train_acc * 100)

    if (!is.null(val_dl)) {
      model$eval()
      val_losses <- c()
      val_correct <- 0
      val_total <- 0

      with_no_grad({
        coro::loop(for (batch in val_dl) {
          x <- batch$x$to(device = device)
          y <- batch$y$to(device = device)
          pred <- model(x)
          loss <- nnf_cross_entropy(pred, y)
          val_losses <- c(val_losses, loss$item())
          predicted <- pred$argmax(dim = 2)
          val_correct <- val_correct + (predicted == y)$sum()$item()
          val_total <- val_total + y$size(1)
        })
      })

      avg_val_loss <- mean(val_losses)
      val_acc <- val_correct / val_total
      history$val_loss <- c(history$val_loss, avg_val_loss)
      history$val_acc <- c(history$val_acc, val_acc)

      msg <- sprintf("%s | Val Loss: %.4f, Val Acc: %.2f%%",
                     msg, avg_val_loss, val_acc * 100)

      if (val_acc > best_val_acc) {
        best_val_acc <- val_acc
        best_val_loss <- avg_val_loss
        torch_save(model, save_path)
        patience_counter <- 0
        msg <- paste0(msg, " *")
      } else {
        patience_counter <- patience_counter + 1
      }

      if (patience_counter >= patience) {
        log_msg("Early stopping at epoch ", epoch)
        break
      }
    } else {
      torch_save(model, save_path)
    }

    if (epoch %% 10 == 0) log_msg(msg)
  }

  log_msg("Training complete. Best Val Acc: ", round(best_val_acc * 100, 1), "%")
  list(model = model, history = history)
}

# ── Prediction ────────────────────────────────────────────────

predict_progression_lstm <- function(model_path, sequences, device = "cpu") {
  model <- torch_load(model_path)
  model <- model$to(device = device)
  model$eval()

  rano_classes <- c("CR", "PR", "SD", "PD")
  predictions <- list()

  with_no_grad({
    for (i in seq_along(sequences)) {
      x <- torch_tensor(sequences[[i]], dtype = torch_float())$unsqueeze(1)
      x <- x$to(device = device)
      logits <- model(x)
      probs <- nnf_softmax(logits, dim = 2)
      pred_class <- logits$argmax(dim = 2)$item() + 1L

      predictions[[i]] <- list(
        class = rano_classes[pred_class],
        class_idx = pred_class,
        probabilities = as.numeric(probs$cpu()),
        confidence = max(as.numeric(probs$cpu()))
      )
    }
  })

  predictions
}
