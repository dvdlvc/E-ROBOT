# Load required packages
library(fields)
library(ggplot2)

# ----------------------------------------------------------------------
# E-ROBOT functions (with centering fix)
# ----------------------------------------------------------------------

truncate_cost <- function(cost_matrix, lambda) {
  pmin(cost_matrix, 2 * lambda)
}

mysinkhorn <- function(a, b, cost_matrix, epsilon, max_iter = 1000, tol = 1e-4) {
  n <- length(a)
  m <- length(b)
  if (!all(dim(cost_matrix) == c(n, m))) {
    stop("Dimension mismatch: cost_matrix must be of size length(a) × length(b)")
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
  u <- as.numeric(u)
  v <- as.numeric(v)
  transport_plan <- K * outer(u, v)
  return(transport_plan)
}

entropic_cost <- function(a, b, cost_matrix, epsilon, max_iter = 1000, tol = 1e-4) {
  pi <- mysinkhorn(a, b, cost_matrix, epsilon, max_iter, tol)
  transport_cost <- sum(pi * cost_matrix)
  entropy <- sum(pi * (log(pi + 1e-10) - log(outer(a, b, "*") + 1e-10)))
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

estimate_location <- function(X, epsilon, lambda, m = 200,
                              max_iter_sinkhorn = 2000, tol_sinkhorn = 1e-6,
                              theta_init = NULL, ...) {
  
  n <- nrow(X)
  d <- ncol(X)
  a <- rep(1/n, n)
  
  # Pre‑generate a fixed standard normal sample and center it
  Z_fixed <- matrix(rnorm(m * d), nrow = m, ncol = d)
  Z_fixed <- scale(Z_fixed, scale = FALSE)   # column means become 0
  
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
  
  if (is.null(theta_init)) {
    theta_init <- colMeans(X)
  }
  
  opt <- optim(par = theta_init,
               fn = loss_fn,
               method = "Nelder-Mead",
               control = list(maxit = 2000, reltol = 1e-8),
               ...)
  
  if (opt$convergence != 0) {
    warning("First optim did not converge. Trying restarts...")
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
    if (best_value == opt$value) {
      warning("All attempts failed to converge. Returning best found.")
    }
    return(best_par)
  } else {
    return(opt$par)
  }
}

# ----------------------------------------------------------------------
# Simulation function for a given contamination level
# ----------------------------------------------------------------------

run_simulation <- function(eta, outlier_value, n, d, true_mean,
                           epsilon, lambda, m,
                           MC.size, max_iter_sinkhorn, tol_sinkhorn) {
  
  # Storage
  theta_sample_mean <- matrix(NA, nrow = MC.size, ncol = d)
  theta_erobot <- matrix(NA, nrow = MC.size, ncol = d)
  
  for (i in 1:MC.size) {
    # Generate clean data
    X_clean <- matrix(rnorm(n * d, mean = true_mean, sd = 1), nrow = n, ncol = d)
    
    # Contaminate if eta > 0
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
    
    # Estimates
    theta_sample_mean[i, ] <- colMeans(X)
    theta_erobot[i, ] <- estimate_location(X, epsilon, lambda, m = m,
                                           max_iter_sinkhorn = max_iter_sinkhorn,
                                           tol_sinkhorn = tol_sinkhorn)
    
    if (i %% 10 == 0) cat("  iteration", i, "done\n")
  }
  
  return(list(mean = theta_sample_mean, erobot = theta_erobot))
}
# ----------------------------------------------------------------------
# Set parameters and scenarios
# ----------------------------------------------------------------------

set.seed(20101978)

n <- 250
d <- 2
true_mean <- rep(0, d)
epsilon <- 1
lambda <- 2.5
m <- n * 2
outlier_value <- 4
MC.size <- 200
max_iter_sinkhorn <- 500
tol_sinkhorn <- 1e-6

# Use character strings for labels; later we'll parse them in facet labels
scenarios <- list(
  clean = list(eta = 0, label = "Clean"),
  eta05 = list(eta = 0.05, label = "eta == 0.05"),
  eta10 = list(eta = 0.15, label = "eta == 0.15")
)

# ----------------------------------------------------------------------
# Run simulations 
# ----------------------------------------------------------------------

results <- list()
for (sc in names(scenarios)) {
  cat("\nRunning scenario:", sc, "\n")
  eta <- scenarios[[sc]]$eta
  res <- run_simulation(eta, outlier_value, n, d, true_mean,
                        epsilon, lambda, m,
                        MC.size, max_iter_sinkhorn, tol_sinkhorn)
  results[[sc]] <- res
}

# ----------------------------------------------------------------------
# Combine results into a single data frame
# ----------------------------------------------------------------------

df_list <- list()
for (sc in names(scenarios)) {
  mean_mat <- results[[sc]]$mean
  erobot_mat <- results[[sc]]$erobot
  
  df_mean <- data.frame(
    value = c(mean_mat[,1], mean_mat[,2]),
    component = rep(c("Theta 1", "Theta 2"), each = MC.size),
    estimator = "MLE",
    scenario = scenarios[[sc]]$label   # character string
  )
  
  df_erobot <- data.frame(
    value = c(erobot_mat[,1], erobot_mat[,2]),
    component = rep(c("Theta 1", "Theta 2"), each = MC.size),
    estimator = "E-ROBOT",
    scenario = scenarios[[sc]]$label   # character string
  )
  
  df_list[[sc]] <- rbind(df_mean, df_erobot)
}

df_all <- do.call(rbind, df_list)

# Convert scenario to factor with desired order
df_all$scenario <- factor(df_all$scenario,
                          levels = c("Clean", "eta == 0.05", "eta == 0.15"))

# ----------------------------------------------------------------------
# Create the combined plot with parsed labels
# ----------------------------------------------------------------------

col_mean <- "grey"
col_erobot <- "dodgerblue"

# Compute symmetric limit
max_abs <- max(abs(df_all$value), na.rm = TRUE)
limit <- max_abs * 1.1   # 10% padding

p <- ggplot(df_all, aes(x = estimator, y = value, fill = estimator)) +
  geom_violin(trim = FALSE, alpha = 1, adjust = 1.5) +
  geom_boxplot(width = 0.15, fill = "white", outlier.shape = NA) +
  stat_summary(fun = "mean", geom = "point", shape = 23, size = 2, fill = "red") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", size = 0.6) +
  facet_grid(scenario ~ component, scales = "fixed",
             labeller = labeller(scenario = label_parsed)) +
  scale_fill_manual(values = c("MLE" = col_mean, "E-ROBOT" = col_erobot)) +
  labs(y = "Estimated value", x = "") +
  coord_cartesian(ylim = c(-limit, limit)) +   # symmetric y-axis
  theme_minimal(base_size = 12) +
  theme(legend.position = "none",
        strip.text = element_text(face = "bold", size = 10),
        axis.text.x = element_text(angle = 0, hjust = 0.5))

print(p)
ggsave("combined_violin_plots.pdf", p, width = 8, height = 8)
ggsave("combined_violin_plots.eps", p, width = 8, height = 8, device = postscript)
# ----------------------------------------------------------------------
# Box plots (instead of violin plots)
# ----------------------------------------------------------------------

col_mean <- "grey"
col_erobot <- "dodgerblue"

# Compute symmetric limit
max_abs <- max(abs(df_all$value), na.rm = TRUE)
limit <- max_abs * 1.1   # add 10% padding

p2 <- ggplot(df_all, aes(x = estimator, y = value, fill = estimator)) +
  geom_boxplot(alpha = 0.8, outlier.shape = 16, outlier.size = 0.5) +
  stat_summary(fun = "mean", geom = "point", shape = 23, size = 2, fill = "red") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", size = 0.6) +
  facet_grid(scenario ~ component, scales = "fixed",
             labeller = labeller(scenario = label_parsed)) +
  scale_fill_manual(values = c("MLE" = col_mean, "E-ROBOT" = col_erobot)) +
  labs(y = "Estimated value", x = "") +
  coord_cartesian(ylim = c(-limit, limit)) +   # symmetric y-axis
  theme_minimal(base_size = 12) +
  theme(legend.position = "none",
        strip.text = element_text(face = "bold", size = 10),
        axis.text.x = element_text(angle = 0, hjust = 0.5))

print(p2)
ggsave("combined_boxplots.pdf", p, width = 8, height = 8)

save(results, df_all, file = "simulation_output.RData")

#load("simulation_output.RData")




# ----------------------------------------------------------------------
# Box plots (instead of violin plots)
# ----------------------------------------------------------------------

col_mean <- "grey"
col_erobot <- "dodgerblue"

# Compute symmetric limit
max_abs <- max(abs(df_all$value), na.rm = TRUE)
limit <- max_abs * 1.1   # add 10% padding

p2 <- ggplot(df_all, aes(x = estimator, y = value, fill = estimator)) +
  geom_boxplot(alpha = 0.8, outlier.shape = 16, outlier.size = 0.5) +
  stat_summary(fun = "mean", geom = "point", shape = 23, size = 2, fill = "red") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", size = 0.6) +
  facet_grid(scenario ~ component, scales = "fixed",
             labeller = labeller(scenario = label_parsed)) +
  scale_fill_manual(values = c("MLE" = col_mean, "E-ROBOT" = col_erobot)) +
  labs(y = "Estimated value", x = "") +
  coord_cartesian(ylim = c(-limit, limit)) +   # symmetric y-axis
  theme_minimal(base_size = 12) +
  theme(legend.position = "none",
        strip.text = element_text(face = "bold", size = 10),
        axis.text.x = element_text(angle = 0, hjust = 0.5))

print(p2)
ggsave("combined_boxplots.pdf", p, width = 8, height = 8)

save(results, df_all, file = "simulation_output.RData")

#load("simulation_output.RData")
# 
# # ----------------------------------------------------------------------
# # Sensitivity curve for E-ROBOT estimator (first component)
# # ----------------------------------------------------------------------
#
estimate_location_deterministic <- function(X, epsilon, lambda, Z_fixed, m = 200,
                                             max_iter_sinkhorn = 500, tol_sinkhorn = 1e-6,
                                             theta_init = NULL) {

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

   if (is.null(theta_init)) {
     theta_init <- colMeans(X)
   }

   opt <- optim(par = theta_init,
                fn = loss_fn,
                method = "Nelder-Mead",
                control = list(maxit = 2000, reltol = 1e-8))

   # Return best even if not converged (add fallback)
   if (opt$convergence != 0) {
     # Try a few restarts
     best_par <- opt$par
     best_value <- opt$value
     for (attempt in 1:5) {
       theta_try <- theta_init + rnorm(d, sd = 0.5)
       opt_try <- optim(par = theta_try,
                        fn = loss_fn,
                        method = "Nelder-Mead",
                        control = list(maxit = 2000, reltol = 1e-8))
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

#
#
#
#

# ----------------------------------------------------------------------
# Sensitivity curve for E-ROBOT (both components)
# ----------------------------------------------------------------------
compute_sensitivity_curve_diagonal <- function(X_base, epsilon, lambda, m = 200,
                                               z_grid = seq(-8, 8, length = 50),
                                               max_iter_sinkhorn = 500, tol_sinkhorn = 1e-6) {

  n <- nrow(X_base)
  d <- ncol(X_base)

  # Pre-generate fixed model samples (deterministic)
  set.seed(1234)
  Z_fixed <- matrix(rnorm(m * d), nrow = m, ncol = d)
  Z_fixed <- scale(Z_fixed, scale = FALSE)

  # Base estimate on full sample
  theta_0 <- estimate_location_deterministic(X_base, epsilon, lambda, Z_fixed,
                                             m = m,
                                             max_iter_sinkhorn = max_iter_sinkhorn,
                                             tol_sinkhorn = tol_sinkhorn)

  SC_values <- matrix(NA, nrow = length(z_grid), ncol = d)

  for (i in seq_along(z_grid)) {
    z <- z_grid[i]

    # Replace the first observation with (z, z) — diagonal contamination
    X_mod <- X_base
    X_mod[1, ] <- c(z, z)   # both coordinates set to z

    theta_z <- estimate_location_deterministic(X_mod, epsilon, lambda, Z_fixed,
                                               m = m,
                                               max_iter_sinkhorn = max_iter_sinkhorn,
                                               tol_sinkhorn = tol_sinkhorn)

    SC_values[i, 1] <- n * (theta_z[1] - theta_0[1])
    SC_values[i, 2] <- n * (theta_z[2] - theta_0[2])

    if (i %% 10 == 0) cat("z =", z, "done\n")
  }

  return(list(z_grid = z_grid, SC = SC_values, theta_0 = theta_0))
}

# ----------------------------------------------------------------------
# Generate base sample (larger n for stability)
# ----------------------------------------------------------------------

set.seed(20101978)
n_base <- 200  # larger sample gives smoother SC
X_base <- matrix(rnorm(n_base * 5, mean = 0, sd = 1), ncol = 2)

z_grid <- seq(-10, 10, length = 50)

SC_data <- compute_sensitivity_curve_diagonal(X_base, epsilon = 1, lambda = 2.5,
                                              m = n_base * 2, z_grid = z_grid)

# Plot both components (they should overlap)
plot(SC_data$z_grid, SC_data$SC[, 1], type = "l", lwd = 2, col = "dodgerblue",
     ylim = c(-2.5, 2.5),
     xlab = expression(z), ylab = "",
     main = "Sensitivity curve")

grid(col = "lightgray", lty = 4, lwd = 2)

# Add component 2 (same color, dashed)
#lines(SC_data$z_grid, SC_data$SC[, 2], lwd = 2, col = "blue", lty = 2)

# MLE theoretical SC: z - mean (for both components)
mean1 <- mean(X_base[, 1])
curve(x - mean1, add = TRUE, col = "grey", lty = 2, lwd = 4)

abline(h = 0, col = "gray", lty = 3)
legend("topleft", legend = c("E-ROBOT M-estimator (Theta1)", "MLE"),
       col = c("dodgerblue", "grey"), lty = c(1, 2), lwd = 3,bty = "n")
#legend("topleft", legend = c("E-ROBOT (comp 1)", "E-ROBOT (comp 2)", "MLE"),
#       col = c("blue", "blue", "grey"), lty = c(1, 2, 1), lwd = 2,bty = "n")
dev.copy2eps(file = "sensitivity_curve.eps", width = 5, height = 7)
dev.off()





# ----------------------------------------------------------------------
# Sensitivity curve for E-ROBOT (both components) with multiple parameter sets
# ----------------------------------------------------------------------

compute_sensitivity_curve_diagonal <- function(X_base, epsilon, lambda, m = 200,
                                               z_grid = seq(-6, 6, length = 20),
                                               max_iter_sinkhorn = 500, tol_sinkhorn = 1e-6) {
  
  n <- nrow(X_base)
  d <- ncol(X_base)
  
  # Pre-generate fixed model samples (deterministic)
  set.seed(1234)
  Z_fixed <- matrix(rnorm(m * d), nrow = m, ncol = d)
  Z_fixed <- scale(Z_fixed, scale = FALSE)
  
  # Base estimate on full sample
  theta_0 <- estimate_location_deterministic(X_base, epsilon, lambda, Z_fixed,
                                             m = m,
                                             max_iter_sinkhorn = max_iter_sinkhorn,
                                             tol_sinkhorn = tol_sinkhorn)
  
  SC_values <- matrix(NA, nrow = length(z_grid), ncol = d)
  
  for (i in seq_along(z_grid)) {
    z <- z_grid[i]
    
    # Replace the first observation with (z, z) — diagonal contamination
    X_mod <- X_base
    X_mod[1, ] <- c(z, z)
    
    theta_z <- estimate_location_deterministic(X_mod, epsilon, lambda, Z_fixed,
                                               m = m,
                                               max_iter_sinkhorn = max_iter_sinkhorn,
                                               tol_sinkhorn = tol_sinkhorn)
    
    SC_values[i, 1] <- n * (theta_z[1] - theta_0[1])
    SC_values[i, 2] <- n * (theta_z[2] - theta_0[2])
    
    if (i %% 10 == 0) cat("z =", z, "done\n")
  }
  
  return(list(z_grid = z_grid, SC = SC_values, theta_0 = theta_0))
}

# ----------------------------------------------------------------------
# Generate base sample (larger n for stability) and different hyperparam
# ----------------------------------------------------------------------

set.seed(20101978)
n_base <- 200  # larger sample gives smoother SC
X_base <- matrix(rnorm(n_base * 5, mean = 0, sd = 1), ncol = 2)

# Reduced grid length for speed (30 points instead of 50)
z_grid <- seq(-6, 6, length = 20)

# Parameter sets to compare
params <- list(
  list(epsilon = 1,   lambda = 2.5, col = "dodgerblue",  lty = 1, label = "E-ROBOT (ε=1, λ=2.5)"),
  list(epsilon = 1,   lambda = 10,  col = "darkorange",  lty = 2, label = "E-ROBOT (ε=1, λ=10)"),
  list(epsilon = 10, lambda = 2.5, col = "forestgreen", lty = 3, label = "E-ROBOT (ε=10, λ=2.5)")
)

# Compute sensitivity curves for each parameter set
SC_list <- list()
for (i in seq_along(params)) {
  cat("\nComputing for", params[[i]]$label, "\n")
  SC_list[[i]] <- compute_sensitivity_curve_diagonal(X_base,
                                                     epsilon = params[[i]]$epsilon,
                                                     lambda  = params[[i]]$lambda,
                                                     m = n_base * 2,
                                                     z_grid = z_grid)
}

# ----------------------------------------------------------------------
# Plot all sensitivity curves together
# ----------------------------------------------------------------------

plot(SC_list[[1]]$z_grid, SC_list[[1]]$SC[, 1],
     type = "l", lwd = 2, col = params[[1]]$col,
     ylim = c(-2.5, 2.5),
     xlab = expression(z), ylab = "",
     main = "Sensitivity curve")

grid(col = "lightgray", lty = 4, lwd = 1)

# Add the other two E-ROBOT curves
for (i in 2:3) {
  lines(SC_list[[i]]$z_grid, SC_list[[i]]$SC[, 1],
        lwd = 2, col = params[[i]]$col, lty = params[[i]]$lty)
}

# MLE theoretical sensitivity curve (unbounded, linear)
mean1 <- mean(X_base[, 1])
curve(x - mean1, add = TRUE, col = "grey", lty = 2, lwd = 4)

# Horizontal line at zero
abline(h = 0, col = "gray", lty = 3)

# Legend
legend("topleft",
       legend = c(params[[1]]$label, params[[2]]$label, params[[3]]$label, "MLE"),
       col = c(params[[1]]$col, params[[2]]$col, params[[3]]$col, "grey"),
       lty = c(params[[1]]$lty, params[[2]]$lty, params[[3]]$lty, 2),
       lwd = 3, bty = "n")
