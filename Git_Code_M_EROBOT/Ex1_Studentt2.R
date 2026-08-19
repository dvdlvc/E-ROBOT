# ----------------------------------------------------------------------
# Multivariate t location: flexible simulation for any two dimensions
# Fixed: no set.seed inside estimate_location_t; Z_fixed pre-generated
# Output: MSE/AbsBias table + kernel density plots for eta=0 and 0.1
# ----------------------------------------------------------------------

library(fields)
library(mvtnorm)
library(tidyr)
library(ggplot2)

# ---------- E-ROBOT functions ----------

truncate_cost <- function(cost_matrix, lambda) {
  pmin(cost_matrix, 2 * lambda)
}

mysinkhorn <- function(a, b, cost_matrix, epsilon, max_iter = 1000, tol = 1e-4) {
  n <- length(a)
  m <- length(b)
  if (!all(dim(cost_matrix) == c(n, m))) {
    stop("Dimension mismatch")
  }
  K <- exp(-cost_matrix / epsilon)
  u <- rep(1, n)
  v <- rep(1, m)
  for (i in 1:max_iter) {
    u_prev <- u
    u <- a / (K %*% v)
    v <- b / (t(K) %*% u)
    if (max(abs(u - u_prev)) < tol) break
  }
  return(K * outer(as.numeric(u), as.numeric(v)))
}

entropic_cost <- function(a, b, cost_matrix, epsilon, max_iter = 1000, tol = 1e-4) {
  pi <- mysinkhorn(a, b, cost_matrix, epsilon, max_iter, tol)
  transport_cost <- sum(pi * cost_matrix)
  eps <- 1e-10
  log_ratio <- log(pmax(pi, eps) / pmax(outer(a, b, "*"), eps))
  entropy <- sum(pi * log_ratio)
  return(transport_cost + epsilon * entropy)
}

truncated_sinkhorn_loss <- function(a, b, cost_matrix, mu_points, nu_points,
                                    epsilon, lambda, max_iter = 1000, tol = 1e-4) {
  cost_matrix_aa <- rdist(mu_points, mu_points)
  cost_matrix_bb <- rdist(nu_points, nu_points)
  c_lambda <- truncate_cost(cost_matrix, lambda)
  c_lambda_aa <- truncate_cost(cost_matrix_aa, lambda)
  c_lambda_bb <- truncate_cost(cost_matrix_bb, lambda)
  
  w_ab <- entropic_cost(a, b, c_lambda, epsilon, max_iter, tol)
  w_aa <- entropic_cost(a, a, c_lambda_aa, epsilon, max_iter, tol)
  w_bb <- entropic_cost(b, b, c_lambda_bb, epsilon, max_iter, tol)
  return(w_ab - 0.5 * (w_aa + w_bb))
}

# ---------- E-ROBOT estimator (now accepts Z_fixed) ----------

estimate_location_t <- function(X, epsilon, lambda, Z_fixed, m = 200,
                                max_iter_sinkhorn = 500, tol_sinkhorn = 1e-6,
                                theta_init = NULL, ...) {
  n <- nrow(X)
  d <- ncol(X)
  a <- rep(1/n, n)
  
  loss_fn <- function(theta) {
    Y <- Z_fixed + matrix(theta, nrow = m, ncol = d, byrow = TRUE)
    b <- rep(1/m, m)
    cost_XY <- rdist(X, Y)
    cost_XX <- rdist(X, X)
    cost_YY <- rdist(Y, Y)
    loss <- truncated_sinkhorn_loss(a, b, cost_XY, X, Y, epsilon, lambda,
                                    max_iter = max_iter_sinkhorn,
                                    tol = tol_sinkhorn)
    return(loss)
  }
  
  if (is.null(theta_init)) theta_init <- colMeans(X)
  opt <- optim(par = theta_init,
               fn = loss_fn,
               method = "Nelder-Mead",
               control = list(maxit = 2000, reltol = 1e-8),
               ...)
  
  if (opt$convergence != 0) {
    best_par <- opt$par
    best_value <- opt$value
    for (attempt in 1:5) {
      theta_try <- theta_init + rnorm(d, sd = 0.5)
      opt_try <- optim(par = theta_try,
                       fn = loss_fn,
                       method = "Nelder-Mead",
                       control = list(maxit = 2000, reltol = 1e-8),
                       ...)
      if (opt_try$convergence == 0 && opt_try$value < best_value) {
        best_par <- opt_try$par
        best_value <- opt_try$value
      }
    }
    return(best_par)
  } else {
    return(opt$par)
  }
}

# ---------- MLE for t-location ----------

mle_t_location <- function(X, df = 2) {
  n <- nrow(X)
  d <- ncol(X)
  neg_log_lik <- function(mu) {
    mu <- as.numeric(mu)
    diff <- X - matrix(mu, nrow = n, ncol = d, byrow = TRUE)
    sqdist <- rowSums(diff^2)
    term <- (df + d) / 2 * sum(log(1 + sqdist / df))
    return(term)
  }
  init <- apply(X, 2, median)
  opt <- optim(par = init,
               fn = neg_log_lik,
               method = "BFGS",
               control = list(maxit = 1000, reltol = 1e-8))
  return(opt$par)
}

# ---------- Simulation function ----------

run_simulation_t <- function(n, eta, true_mean, d, Z_fixed,
                             epsilon, lambda, m,
                             MC.size = 100,
                             max_iter_sinkhorn = 500, tol_sinkhorn = 1e-6,
                             outlier_value = 120) {
  
  mle_est <- matrix(NA, nrow = MC.size, ncol = d)
  erobot_est <- matrix(NA, nrow = MC.size, ncol = d)
  
  pb <- txtProgressBar(min = 0, max = MC.size, style = 3)
  for (rep in 1:MC.size) {
    X_clean <- rmvt(n, sigma = diag(d), df = 2)
    X_clean <- sweep(X_clean, 2, true_mean, "+")
    if (eta > 0) {
      n_out <- round(eta * n)
      if (n_out > 0) {
        idx <- sample(1:n, size = n_out, replace = FALSE)
        X <- X_clean
        X[idx, ] <- outlier_value
      } else {
        X <- X_clean
      }
    } else {
      X <- X_clean
    }
    mle_est[rep, ] <- mle_t_location(X, df = 2)
    erobot_est[rep, ] <- estimate_location_t(X, epsilon, lambda, Z_fixed, m = m,
                                             max_iter_sinkhorn = max_iter_sinkhorn,
                                             tol_sinkhorn = tol_sinkhorn)
    setTxtProgressBar(pb, rep)
  }
  close(pb)
  
  return(list(mle_est = mle_est, erobot_est = erobot_est))
}

# ---------- Main simulation ----------

# SET DIMENSIONS HERE (two values)
dimensions <- c(2, 4)   # change as needed

if (length(dimensions) != 2) {
  stop("Please provide exactly two dimensions.")
}

# Parameters
n <- 60
epsilon <- 2
lambda <- 3
m <- n * 8
MC.size <- 200      # increase to 100 for final
outlier_value <- -20
eta_values <- c(0, 0.05, 0.1)

# Set global seed for data generation
set.seed(20101978)

# Pre-generate Z_fixed for each dimension (different seed)
Z_fixed_list <- list()
for (d in dimensions) {
  set.seed(12345)  # fixed seed for model samples only
  Z_fixed <- rmvt(m, sigma = diag(d), df = 2)
  Z_fixed_list[[paste0("d", d)]] <- Z_fixed
}

# Storage
all_results <- list()
summary_df <- data.frame()

for (d in dimensions) {
  true_mean <- rep(0, d)
  Z_fixed <- Z_fixed_list[[paste0("d", d)]]
  for (eta in eta_values) {
    cat("\n=== d =", d, ", eta =", eta, "===\n")
    res <- run_simulation_t(n, eta, true_mean, d, Z_fixed,
                            epsilon, lambda, m, MC.size,
                            outlier_value = outlier_value)
    
    # Compute metrics for MLE
    mle_mat <- res$mle_est
    sq_err <- rowSums((mle_mat - matrix(true_mean, nrow = MC.size, ncol = d, byrow = TRUE))^2)
    mse_mle <- mean(sq_err)
    abs_err <- rowMeans(abs(mle_mat - matrix(true_mean, nrow = MC.size, ncol = d, byrow = TRUE)))
    abs_bias_mle <- mean(abs_err)
    
    # Compute metrics for E-ROBOT
    erobot_mat <- res$erobot_est
    sq_err_er <- rowSums((erobot_mat - matrix(true_mean, nrow = MC.size, ncol = d, byrow = TRUE))^2)
    mse_erobot <- mean(sq_err_er)
    abs_err_er <- rowMeans(abs(erobot_mat - matrix(true_mean, nrow = MC.size, ncol = d, byrow = TRUE)))
    abs_bias_erobot <- mean(abs_err_er)
    
    all_results[[paste0("d", d, "_eta", eta)]] <- res
    
    row_mle <- data.frame(d = d, eta = eta, estimator = "MLE",
                          MSE = mse_mle, AbsBias = abs_bias_mle)
    row_erobot <- data.frame(d = d, eta = eta, estimator = "E-ROBOT",
                             MSE = mse_erobot, AbsBias = abs_bias_erobot)
    summary_df <- rbind(summary_df, row_mle, row_erobot)
  }
}

save(all_results, summary_df, file = "t_location_simulation_all.RData")
cat("Results saved to t_location_simulation_all.RData\n")
# ---------- Recompute summary statistics after clipping extreme estimates ----------

clip_threshold <- 10   # exclude estimates with norm > 10

# Storage for new summary
summary_df <- data.frame()

for (d in dimensions) {
  true_mean <- rep(0, d)
  for (eta in eta_values) {
    key <- paste0("d", d, "_eta", eta)
    res <- all_results[[key]]
    if (is.null(res)) next
    
    mle_mat <- res$mle_est
    erobot_mat <- res$erobot_est
    
    # --- MLE: filter by norm ---
    mle_norm <- sqrt(rowSums(mle_mat^2))
    keep_mle <- which(mle_norm <= clip_threshold)
    if (length(keep_mle) > 0) {
      mle_sub <- mle_mat[keep_mle, , drop = FALSE]
      sq_err <- rowSums((mle_sub - matrix(true_mean, nrow = nrow(mle_sub), ncol = d, byrow = TRUE))^2)
      mse_mle <- mean(sq_err)
      abs_err <- rowMeans(abs(mle_sub - matrix(true_mean, nrow = nrow(mle_sub), ncol = d, byrow = TRUE)))
      abs_bias_mle <- mean(abs_err)
      cat("d=", d, " eta=", eta, " MLE kept:", length(keep_mle), "/", nrow(mle_mat), "\n")
    } else {
      mse_mle <- NA
      abs_bias_mle <- NA
      cat("d=", d, " eta=", eta, " MLE: all estimates excluded!\n")
    }
    
    # --- E-ROBOT: filter by norm ---
    erobot_norm <- sqrt(rowSums(erobot_mat^2))
    keep_erobot <- which(erobot_norm <= clip_threshold)
    if (length(keep_erobot) > 0) {
      erobot_sub <- erobot_mat[keep_erobot, , drop = FALSE]
      sq_err_er <- rowSums((erobot_sub - matrix(true_mean, nrow = nrow(erobot_sub), ncol = d, byrow = TRUE))^2)
      mse_erobot <- mean(sq_err_er)
      abs_err_er <- rowMeans(abs(erobot_sub - matrix(true_mean, nrow = nrow(erobot_sub), ncol = d, byrow = TRUE)))
      abs_bias_erobot <- mean(abs_err_er)
      cat("d=", d, " eta=", eta, " E-ROBOT kept:", length(keep_erobot), "/", nrow(erobot_mat), "\n")
    } else {
      mse_erobot <- NA
      abs_bias_erobot <- NA
      cat("d=", d, " eta=", eta, " E-ROBOT: all estimates excluded!\n")
    }
    
    # Append to summary
    row_mle <- data.frame(d = d, eta = eta, estimator = "MLE",
                          MSE = mse_mle, AbsBias = abs_bias_mle)
    row_erobot <- data.frame(d = d, eta = eta, estimator = "E-ROBOT",
                             MSE = mse_erobot, AbsBias = abs_bias_erobot)
    summary_df <- rbind(summary_df, row_mle, row_erobot)
  }
}

# ---------- LaTeX table (now based on filtered summary) ----------

# Reshape into wide format
wide_df <- summary_df %>%
  pivot_wider(id_cols = c(estimator, eta),
              names_from = d,
              values_from = c(MSE, AbsBias),
              names_sep = "_d")

# Reorder rows: E-ROBOT first, then MLE
wide_df$estimator <- factor(wide_df$estimator, levels = c("E-ROBOT", "MLE"))
wide_df <- wide_df[order(wide_df$estimator, wide_df$eta), ]

fmt <- function(x) sprintf("%.4f", x)

d1 <- dimensions[1]
d2 <- dimensions[2]

MSE_col1 <- paste0("MSE_d", d1)
Abs_col1 <- paste0("AbsBias_d", d1)
MSE_col2 <- paste0("MSE_d", d2)
Abs_col2 <- paste0("AbsBias_d", d2)

cat("\n\\begin{table}[htbp]\n\\centering\n")
cat("\\caption{Comparison of MLE and E-ROBOT for multivariate $t_2$ location (n=", n, "). 
    MSE is the mean squared Euclidean distance; AbsBias is the mean absolute error across components. 
    Outliers: observations replaced by ", outlier_value, " in all dimensions. 
    True location is zero. Results based on ", MC.size, " Monte Carlo repetitions, 
    after excluding estimates with norm $>", clip_threshold, "$.}\n")
cat("\\begin{tabular}{lccc|ccc}\n")
cat("\\hline\n")
cat("Estimator & \\multicolumn{3}{c|}{\\textbf{d=", d1, "}} & \\multicolumn{2}{c}{\\textbf{d=", d2, "}} \\\\ \n")
cat("\\cline{2-6}\n")
cat(" & \\eta & MSE & AbsBias & MSE & AbsBias \\\\ \n")
cat("\\hline \\hline\n")

for (i in 1:nrow(wide_df)) {
  row <- wide_df[i, ]
  est <- as.character(row$estimator)
  eta <- row$eta
  mse1 <- row[[MSE_col1]]
  abs1 <- row[[Abs_col1]]
  mse2 <- row[[MSE_col2]]
  abs2 <- row[[Abs_col2]]
  eta_label <- ifelse(eta == 0, "0", paste0(eta*100, "\\%"))
  cat(est, "  &  ", eta_label, "  &  ",
      fmt(mse1), "  &  ", fmt(abs1), "  &  ",
      fmt(mse2), "  &  ", fmt(abs2), " \\\\ \n")
}
cat("\\hline \\hline\n")
cat("\\end{tabular}\n")
cat("\\label{tab:t_location_summary}\n")
cat("\\end{table}\n")

# Save table to file
sink("t_location_summary_table.tex")
cat("\\begin{table}[htbp]\n\\centering\n")
cat("\\caption{Comparison of MLE and E-ROBOT for multivariate $t_2$ location (n=", n, "). 
    MSE is the mean squared Euclidean distance; AbsBias is the mean absolute error across components. 
    Outliers: observations replaced by ", outlier_value, " in all dimensions. 
    True location is zero. Results based on ", MC.size, " Monte Carlo repetitions, 
    after excluding estimates with norm $>", clip_threshold, "$.}\n")
cat("\\begin{tabular}{lccc|ccc}\n")
cat("\\hline\n")
cat("Estimator & \\multicolumn{3}{c|}{\\textbf{d=", d1, "}} & \\multicolumn{2}{c}{\\textbf{d=", d2, "}} \\\\ \n")
cat("\\cline{2-6}\n")
cat(" & \\eta & MSE & AbsBias & MSE & AbsBias \\\\ \n")
cat("\\hline \\hline\n")
for (i in 1:nrow(wide_df)) {
  row <- wide_df[i, ]
  est <- as.character(row$estimator)
  eta <- row$eta
  mse1 <- row[[MSE_col1]]
  abs1 <- row[[Abs_col1]]
  mse2 <- row[[MSE_col2]]
  abs2 <- row[[Abs_col2]]
  eta_label <- ifelse(eta == 0, "0", paste0(eta*100, "\\%"))
  cat(est, "  &  ", eta_label, "  &  ",
      fmt(mse1), "  &  ", fmt(abs1), "  &  ",
      fmt(mse2), "  &  ", fmt(abs2), " \\\\ \n")
}
cat("\\hline \\hline\n")
cat("\\end{tabular}\n")
cat("\\label{tab:t_location_summary}\n")
cat("\\end{table}\n")
sink()

cat("Table saved to t_location_summary_table.tex\n")


# ---------- Kernel density plots for first component at eta=0 and 0.1 (only d=4) ----------
# ---------- Kernel density plots for first component at eta=0 and 0.1 (only d=4) ----------

library(ggplot2)

# Extract data for eta=0 and eta=0.1, but only for d=4
dens_data <- list()
d_keep <- 4   # <--- change to the dimension you want to plot
for (eta in c(0, 0.1)) {
  key <- paste0("d", d_keep, "_eta", eta)
  res <- all_results[[key]]
  if (!is.null(res)) {
    mle_vec <- res$mle_est[, 1]
    erobot_vec <- res$erobot_est[, 1]
    df_mle <- data.frame(value = mle_vec, estimator = "MLE", eta = eta, dimension = paste0("d = ", d_keep))
    df_erobot <- data.frame(value = erobot_vec, estimator = "E-ROBOT", eta = eta, dimension = paste0("d = ", d_keep))
    dens_data[[paste0("eta", eta)]] <- rbind(df_mle, df_erobot)
  }
}
df_dens <- do.call(rbind, dens_data)

# Convert eta to factor for faceting
df_dens$eta <- factor(df_dens$eta, levels = c(0, 0.1))

# Create the density plot with proper expansion (no coordinate clipping)
p_dens <- ggplot(df_dens, aes(x = value, fill = estimator, colour = estimator)) +
  geom_density(alpha = 0.5, adjust = 2, size = 0.8) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black", size = 0.6) +
  facet_wrap(~ eta, ncol = 2, labeller = label_bquote(eta == .(as.character(eta)))) +
  scale_fill_manual(values = c("MLE" = "grey", "E-ROBOT" = "dodgerblue")) +
  scale_colour_manual(values = c("MLE" = "grey40", "E-ROBOT" = "dodgerblue4")) +
  scale_x_continuous(expand = expansion(mult = 0.3)) +   # add 30% padding to show full tails
  labs(title = paste("Kernel density of first component estimates (d =", d_keep, ")"),
       x = "Estimated value", y = "Density") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom",
        strip.text = element_text(face = "bold", size = 12),
        legend.title = element_blank())

print(p_dens)

# Save the density plot
ggsave("density_eta0_eta01_d4.pdf", p_dens, width = 8, height = 4)
ggsave("density_eta0_eta01_d4.eps", p_dens, width = 8, height = 4, device = cairo_ps)

cat("Density plot saved to density_eta0_eta01_d4.pdf and .eps\n")



# ---------- Violin plot for first component at eta=0 and eta=0.1 (only d=4) ----------

library(ggplot2)

# Set dimension
d_keep <- 4   # change to your desired dimension

# Extract data for eta=0 and eta=0.1 for the chosen dimension
violin_data <- list()
eta_values_plot <- c(0, 0.1)

for (eta in eta_values_plot) {
  key <- paste0("d", d_keep, "_eta", eta)
  res <- all_results[[key]]
  if (!is.null(res)) {
    mle_vec <- res$mle_est[, 1]
    erobot_vec <- res$erobot_est[, 1]
    df_mle <- data.frame(value = mle_vec, estimator = "MLE", eta = as.factor(eta))
    df_erobot <- data.frame(value = erobot_vec, estimator = "E-ROBOT", eta = as.factor(eta))
    violin_data[[paste0("eta", eta)]] <- rbind(df_mle, df_erobot)
  }
}
df_violin <- do.call(rbind, violin_data)

# Create the violin plot
p_violin <- ggplot(df_violin, aes(x = estimator, y = value, fill = estimator)) +
  geom_violin(trim = FALSE, alpha = 0.8, adjust = 1.5) +
  geom_boxplot(width = 0.15, fill = "white", outlier.shape = NA) +
  stat_summary(fun = "mean", geom = "point", shape = 23, size = 2, fill = "red") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", size = 0.6) +
  facet_wrap(~ eta, ncol = 2, labeller = label_bquote(eta == .(as.character(eta)))) +
  scale_fill_manual(values = c("MLE" = "grey", "E-ROBOT" = "dodgerblue")) +
  labs(title = paste("First component estimates (d =", d_keep, ")"),
       y = "Estimated value", x = "") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom",
        strip.text = element_text(face = "bold", size = 12),
        legend.title = element_blank())

print(p_violin)

# Save
ggsave("violin_eta0_eta01_d4.pdf", p_violin, width = 8, height = 4)
ggsave("violin_eta0_eta01_d4.eps", p_violin, width = 8, height = 4, device = cairo_ps)

cat("Violin plot saved to violin_eta0_eta01_d4.pdf and .eps\n")




library(ggplot2)

# Set dimension
d_keep <- 4   # change to your desired dimension

# Extract data for eta=0 and eta=0.1 for the chosen dimension, for the SECOND component
violin_data <- list()
eta_values_plot <- c(0, 0.1)

for (eta in eta_values_plot) {
  key <- paste0("d", d_keep, "_eta", eta)
  res <- all_results[[key]]
  if (!is.null(res)) {
    mle_vec <- res$mle_est[, 2]      # <--- changed to second component
    erobot_vec <- res$erobot_est[, 2] # <--- changed to second component
    df_mle <- data.frame(value = mle_vec, estimator = "MLE", eta = as.factor(eta))
    df_erobot <- data.frame(value = erobot_vec, estimator = "E-ROBOT", eta = as.factor(eta))
    violin_data[[paste0("eta", eta)]] <- rbind(df_mle, df_erobot)
  }
}
df_violin <- do.call(rbind, violin_data)

# Create the violin plot
p_violin <- ggplot(df_violin, aes(x = estimator, y = value, fill = estimator)) +
  geom_violin(trim = FALSE, alpha = 0.8, adjust = 1.5) +
  geom_boxplot(width = 0.15, fill = "white", outlier.shape = NA) +
  stat_summary(fun = "mean", geom = "point", shape = 23, size = 2, fill = "red") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", size = 0.6) +
  facet_wrap(~ eta, ncol = 2, labeller = label_bquote(eta == .(as.character(eta)))) +
  scale_fill_manual(values = c("MLE" = "grey", "E-ROBOT" = "dodgerblue")) +
  labs(title = paste("Second component estimates (d =", d_keep, ")"),
       y = "Estimated value", x = "") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom",
        strip.text = element_text(face = "bold", size = 12),
        legend.title = element_blank())

print(p_violin)

# Save
ggsave("violin_eta0_eta01_d4_comp2.pdf", p_violin, width = 8, height = 4)
ggsave("violin_eta0_eta01_d4_comp2.eps", p_violin, width = 8, height = 4, device = cairo_ps)

cat("Violin plot for second component saved to violin_eta0_eta01_d4_comp2.pdf and .eps\n")


# ---------- Kernel density plots using base R (no ggplot) ----------

# Set the dimension to plot
d_keep <- 4   # change to 2 if needed

# Extract data for eta=0 and eta=0.1 for the chosen dimension
dens_data <- list()
eta_values_plot <- c(0, 0.1)

for (eta in eta_values_plot) {
  key <- paste0("d", d_keep, "_eta", eta)
  res <- all_results[[key]]
  if (!is.null(res)) {
    mle_vec <- res$mle_est[, 1]
    erobot_vec <- res$erobot_est[, 1]
    dens_data[[paste0("eta", eta)]] <- list(
      mle = mle_vec,
      erobot = erobot_vec
    )
  } else {
    warning("Results not found for d = ", d_keep, " eta = ", eta)
    next
  }
}

# Compute density objects for each group
dens_objects <- list()
all_x <- c()
all_y <- c()

for (eta in eta_values_plot) {
  if (is.null(dens_data[[paste0("eta", eta)]])) next
  mle_vec <- dens_data[[paste0("eta", eta)]]$mle
  erobot_vec <- dens_data[[paste0("eta", eta)]]$erobot
  
  # Compute densities (adjust bandwidth if needed, here default)
  d_mle <- density(mle_vec, na.rm = TRUE, bw=0.25)
  d_erobot <- density(erobot_vec, na.rm = TRUE,bw=0.25)
  
  dens_objects[[paste0("eta", eta, "_mle")]] <- d_mle
  dens_objects[[paste0("eta", eta, "_erobot")]] <- d_erobot
  
  # Collect x and y ranges for consistent axis
  all_x <- c(all_x, d_mle$x, d_erobot$x)
  all_y <- c(all_y, d_mle$y, d_erobot$y)
}

if (length(all_x) == 0) stop("No data to plot.")

# Determine axis limits with padding
xlim <- range(all_x)
xlim <- xlim + c(-0.3, 0.3) * diff(xlim)   # add 30% padding on both sides
ylim <- range(c(0, all_y))
ylim[2] <- ylim[2] * 1.1   # add 10% top padding

# Set up two panels side by side
par(mfrow = c(1, 2), mar = c(5, 4, 4, 2) + 0.1)

for (eta in eta_values_plot) {
  key_mle <- paste0("eta", eta, "_mle")
  key_erobot <- paste0("eta", eta, "_erobot")
  
  if (!(key_mle %in% names(dens_objects)) || !(key_erobot %in% names(dens_objects))) {
    next
  }
  
  d_mle <- dens_objects[[key_mle]]
  d_erobot <- dens_objects[[key_erobot]]
  
  # Create empty plot with correct axes
  plot(d_mle$x, d_mle$y, type = "n",
       xlim = xlim, ylim = ylim,
       xlab = "Estimated value", ylab = "Density",
       main = bquote(eta == .(eta)))
  
  # Add density lines
  lines(d_mle, col = "grey40", lwd = 3, lty = 2)      # MLE grey dashed
  lines(d_erobot, col = "dodgerblue", lwd = 3, lty = 1)  # E-ROBOT blue solid
  
  # Add vertical line at zero
  abline(v = 0, lty = 2, col = "black")
  
  # Add legend only for the first panel, one line
  #if (eta == eta_values_plot[1]) {
  #  legend(x=-2, y=0.6, legend = c("MLE", "E-ROBOT"),
  #         col = c("black", "dodgerblue"), lwd = 1, lty = c(2, 1),
  #         bty = "o", bg = "white", ncol = 2)   # box with white background
  #}
}

# Save as PDF and EPS
dev.copy2pdf(file = "density_eta0_eta01_d4_baseR.pdf", width = 10, height = 5)
dev.copy2eps(file = "density_eta0_eta01_d4_baseR.eps", width = 10, height = 5)

# Reset graphics parameters
par(mfrow = c(1, 1))

cat("Density plots saved to density_eta0_eta01_d4_baseR.pdf and .eps\n")