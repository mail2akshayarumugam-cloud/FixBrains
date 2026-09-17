# ──────────────────────────────────────────────────────────────
# Stage 3: 3D U-Net for Tumor Segmentation (Pure R torch)
# ──────────────────────────────────────────────────────────────

library(torch)
library(luz)

# ── 3D Convolutional Block ────────────────────────────────────

conv_block_3d <- nn_module(
  "ConvBlock3D",
  initialize = function(in_channels, out_channels) {
    self$conv1 <- nn_conv3d(in_channels, out_channels, kernel_size = 3, padding = 1)
    self$bn1   <- nn_batch_norm3d(out_channels)
    self$conv2 <- nn_conv3d(out_channels, out_channels, kernel_size = 3, padding = 1)
    self$bn2   <- nn_batch_norm3d(out_channels)
    self$relu  <- nn_relu()
  },
  forward = function(x) {
    x <- self$relu(self$bn1(self$conv1(x)))
    x <- self$relu(self$bn2(self$conv2(x)))
    x
  }
)

# ── Encoder Block ─────────────────────────────────────────────

encoder_block_3d <- nn_module(
  "EncoderBlock3D",
  initialize = function(in_channels, out_channels) {
    self$conv_block <- conv_block_3d(in_channels, out_channels)
    self$pool <- nn_max_pool3d(kernel_size = 2, stride = 2)
  },
  forward = function(x) {
    features <- self$conv_block(x)
    pooled <- self$pool(features)
    list(features = features, pooled = pooled)
  }
)

# ── Decoder Block ─────────────────────────────────────────────

decoder_block_3d <- nn_module(
  "DecoderBlock3D",
  initialize = function(in_channels, out_channels) {
    self$upconv <- nn_conv_transpose3d(in_channels, out_channels,
                                        kernel_size = 2, stride = 2)
    self$conv_block <- conv_block_3d(out_channels * 2, out_channels)
  },
  forward = function(x, skip) {
    x <- self$upconv(x)

    diff_d <- skip$size(3) - x$size(3)
    diff_h <- skip$size(4) - x$size(4)
    diff_w <- skip$size(5) - x$size(5)
    x <- nnf_pad(x, c(
      diff_w %/% 2, diff_w - diff_w %/% 2,
      diff_h %/% 2, diff_h - diff_h %/% 2,
      diff_d %/% 2, diff_d - diff_d %/% 2
    ))

    x <- torch_cat(list(x, skip), dim = 2)
    self$conv_block(x)
  }
)

# ── Full 3D U-Net ─────────────────────────────────────────────

unet_3d <- nn_module(
  "UNet3D",
  initialize = function(in_channels = 4L, n_classes = 4L, base_filters = 16L) {
    f <- base_filters

    self$enc1 <- encoder_block_3d(in_channels, f)
    self$enc2 <- encoder_block_3d(f, f * 2L)
    self$enc3 <- encoder_block_3d(f * 2L, f * 4L)
    self$enc4 <- encoder_block_3d(f * 4L, f * 8L)

    self$bottleneck <- conv_block_3d(f * 8L, f * 16L)

    self$dec4 <- decoder_block_3d(f * 16L, f * 8L)
    self$dec3 <- decoder_block_3d(f * 8L, f * 4L)
    self$dec2 <- decoder_block_3d(f * 4L, f * 2L)
    self$dec1 <- decoder_block_3d(f * 2L, f)

    self$final_conv <- nn_conv3d(f, n_classes, kernel_size = 1)
  },
  forward = function(x) {
    e1 <- self$enc1(x)
    e2 <- self$enc2(e1$pooled)
    e3 <- self$enc3(e2$pooled)
    e4 <- self$enc4(e3$pooled)

    b <- self$bottleneck(e4$pooled)

    d4 <- self$dec4(b, e4$features)
    d3 <- self$dec3(d4, e3$features)
    d2 <- self$dec2(d3, e2$features)
    d1 <- self$dec1(d2, e1$features)

    self$final_conv(d1)
  }
)

# ── Dice Loss for Segmentation ────────────────────────────────

dice_loss <- function(pred, target, smooth = 1.0) {
  pred_soft <- nnf_softmax(pred, dim = 2)
  n_classes <- pred$size(2)
  total_loss <- torch_tensor(0, device = pred$device)

  for (c in seq_len(n_classes)) {
    pred_c <- pred_soft[, c, , , ]
    target_c <- (target == (c - 1L))$to(dtype = torch_float())

    intersection <- (pred_c * target_c)$sum()
    union <- pred_c$sum() + target_c$sum()

    dice <- (2 * intersection + smooth) / (union + smooth)
    total_loss <- total_loss + (1 - dice)
  }

  total_loss / n_classes
}

combined_loss <- function(pred, target) {
  ce <- nnf_cross_entropy(pred, target$to(dtype = torch_long()))
  dl <- dice_loss(pred, target)
  0.5 * ce + 0.5 * dl
}

# ── Dataset for Training ──────────────────────────────────────

tumor_seg_dataset <- dataset(
  "TumorSegDataset",
  initialize = function(mri_paths, seg_paths) {
    self$mri_paths <- mri_paths
    self$seg_paths <- seg_paths
    ext <- tools::file_ext(mri_paths[1])
    self$is_pt <- tolower(ext) == "pt"
  },
  .getitem = function(i) {
    if (self$is_pt) {
      mri_tensor <- torch_load(self$mri_paths[i])$to(dtype = torch_float())
      seg_tensor <- torch_load(self$seg_paths[i])$to(dtype = torch_long())
    } else {
      mri <- readRDS(self$mri_paths[i])
      seg <- readRDS(self$seg_paths[i])
      mri_tensor <- torch_tensor(mri, dtype = torch_float())
      seg_tensor <- torch_tensor(seg, dtype = torch_long())
    }

    if (mri_tensor$dim() == 3) {
      mri_tensor <- mri_tensor$unsqueeze(1)
    } else if (mri_tensor$dim() == 4 && mri_tensor$size(4) <= 4) {
      mri_tensor <- mri_tensor$permute(c(4, 1, 2, 3))
    }

    list(x = mri_tensor, y = seg_tensor)
  },
  .length = function() {
    length(self$mri_paths)
  }
)

# ── Training Function ─────────────────────────────────────────

train_unet <- function(train_mri_paths, train_seg_paths,
                       val_mri_paths = NULL, val_seg_paths = NULL,
                       epochs = 50L, batch_size = 2L, lr = 1e-4,
                       in_channels = 4L, n_classes = 4L,
                       base_filters = 16L, save_path = "models/unet_3d.pt") {

  device <- if (cuda_is_available()) "cuda" else "cpu"
  log_msg("Training on device: ", device)

  train_ds <- tumor_seg_dataset(train_mri_paths, train_seg_paths)
  train_dl <- dataloader(train_ds, batch_size = batch_size, shuffle = TRUE)

  val_dl <- NULL
  if (!is.null(val_mri_paths)) {
    val_ds <- tumor_seg_dataset(val_mri_paths, val_seg_paths)
    val_dl <- dataloader(val_ds, batch_size = batch_size, shuffle = FALSE)
  }

  model <- unet_3d(in_channels, n_classes, base_filters)
  model <- model$to(device = device)
  optimizer <- optim_adam(model$parameters, lr = lr)

  best_loss <- Inf
  history <- list(train_loss = numeric(0), val_loss = numeric(0))

  for (epoch in seq_len(epochs)) {
    model$train()
    train_losses <- c()

    coro::loop(for (batch in train_dl) {
      x <- batch$x$to(device = device)
      y <- batch$y$to(device = device)

      optimizer$zero_grad()
      pred <- model(x)
      loss <- combined_loss(pred, y)
      loss$backward()
      optimizer$step()

      train_losses <- c(train_losses, loss$item())
    })

    avg_train <- mean(train_losses)
    history$train_loss <- c(history$train_loss, avg_train)

    msg <- sprintf("Epoch %d/%d — Train Loss: %.4f", epoch, epochs, avg_train)

    if (!is.null(val_dl)) {
      model$eval()
      val_losses <- c()

      with_no_grad({
        coro::loop(for (batch in val_dl) {
          x <- batch$x$to(device = device)
          y <- batch$y$to(device = device)
          pred <- model(x)
          loss <- combined_loss(pred, y)
          val_losses <- c(val_losses, loss$item())
        })
      })

      avg_val <- mean(val_losses)
      history$val_loss <- c(history$val_loss, avg_val)
      msg <- sprintf("%s | Val Loss: %.4f", msg, avg_val)

      if (avg_val < best_loss) {
        best_loss <- avg_val
        torch_save(model, save_path)
        msg <- sprintf("%s *", msg)
      }
    } else {
      if (avg_train < best_loss) {
        best_loss <- avg_train
        torch_save(model, save_path)
      }
    }

    log_msg(msg)
  }

  log_msg("Training complete. Best loss: ", round(best_loss, 4))
  list(model = model, history = history)
}

# ── Inference ─────────────────────────────────────────────────

predict_segmentation <- function(model_path, mri_stack, device = "cpu") {
  model <- torch_load(model_path)
  model <- model$to(device = device)
  model$eval()

  mri_tensor <- torch_tensor(mri_stack, dtype = torch_float())
  if (mri_tensor$dim() == 3) {
    mri_tensor <- mri_tensor$unsqueeze(1)$unsqueeze(1)
  } else if (mri_tensor$dim() == 4) {
    mri_tensor <- mri_tensor$permute(c(4, 1, 2, 3))$unsqueeze(1)
  }
  mri_tensor <- mri_tensor$to(device = device)

  with_no_grad({
    pred <- model(mri_tensor)
    pred_class <- pred$argmax(dim = 2)$squeeze(1)
  })

  as.array(pred_class$cpu())
}
