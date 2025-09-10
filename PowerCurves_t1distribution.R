

library(Matrix)
library(transport)  # For optimal transport utilities
library(fields)
library("mvtnorm")
set.seed(123)
# Truncated cost function
truncate_cost <- function(cost_matrix, lambda) {
  pmin(cost_matrix, 2 * lambda)
}

t_df = 1

# Sinkhorn algorithm with entropic regularization
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
  # Ensure u and v are numeric vectors
  u <- as.numeric(u)
  v <- as.numeric(v)
  
  
  transport_plan <- K * outer(u, v)
  return(transport_plan)
}


# Compute entropic OT cost
entropic_cost <- function(a, b, cost_matrix, epsilon) {
  pi <- mysinkhorn(a, b, cost_matrix, epsilon)
  transport_cost <- sum(pi * cost_matrix)
  entropy <- sum(pi * (log(pi + 1e-10) - log(outer(a, b, "*") + 1e-10)))
  return(transport_cost + epsilon * entropy)
}

# Truncated Sinkhorn loss
truncated_sinkhorn_loss <- function(a, b, cost_matrix, mu_points, nu_points, epsilon, lambda) {
  
  cost_matrix_aa <- rdist(mu_points, mu_points)  # Shape: (n_mu, n_mu)
  cost_matrix_bb <- rdist(nu_points, nu_points)  # Shape: (n_nu, n_nu)
  c_lambda <- truncate_cost(cost_matrix, lambda)
  c_lambda_aa <- truncate_cost(cost_matrix_aa, lambda)
  c_lambda_bb <- truncate_cost(cost_matrix_bb, lambda)
  
  w_ab <- entropic_cost(a, b, c_lambda, epsilon)
  w_aa <- entropic_cost(a, a, c_lambda_aa, epsilon)
  w_bb <- entropic_cost(b, b, c_lambda_bb, epsilon)
  return(w_ab - 0.5 * (w_aa + w_bb))
}




simulate_Tn_null <- function(n_mu, epsilon, lambda, mydim, J, seed=123) {
  # Define different number of points and the setting
  set.seed=seed
  n_nu <- n_mu * 5
  mean1<- 0
  mean_ref<-0
  mean_mu = rep(mean1,mydim)
  mean_nu = rep(mean_ref,mydim)
  mySigma<- diag(mydim)
  Tn<-as.vector(rep(0,J))
  for (j in 1:J) {
    # Generate the point clouds
    mu_points <- rmvt(n_mu, sigma = mySigma, df = t_df, delta = mean_mu)
    nu_points <- rmvt(n_nu, sigma = mySigma, df = t_df, delta = mean_nu)
    cost_matrix <- rdist(mu_points, nu_points)  
    a <- rep(1 / n_mu, n_mu)
    b <- rep(1 / n_nu, n_nu)
    loss <- truncated_sinkhorn_loss(a, b, cost_matrix, mu_points, nu_points, epsilon, lambda)
    Tn[j]<-loss
    print(j)
  }
  return(Tn)
}

simulate_W_null <- function(n_mu, mydim, J, seed=123) {
  # Define different number of points and the setting
  set.seed=seed
  n_nu <- n_mu * 5
  mean1<- 0
  mean_ref<-0
  mean_mu = rep(mean1,mydim)
  mean_nu = rep(mean_ref,mydim)
  mySigma<- diag(mydim)
  Wn<-as.vector(rep(0,J))
  for (j in 1:J) {
    # Generate the point clouds
    mu_points <- rmvt(n_mu, sigma = mySigma, df = t_df, delta = mean_mu)
    nu_points <- rmvt(n_nu, sigma = mySigma, df = t_df, delta = mean_nu)
    cost_matrix <- rdist(mu_points, nu_points)  
    a <- rep(1 / n_mu, n_mu)
    b <- rep(1 / n_nu, n_nu)
    Wn[j]<-transport::wasserstein(a,b,costm=cost_matrix, p=1)
    print(j)
  }
  return(Wn)
}




######################
# d varying
######################
myJ=300
mylambda = 3
n_mu = 50
epsilon = 0.05



myd=25 #dimension
T_null_15<-simulate_Tn_null(n_mu, epsilon=epsilon, lambda=mylambda, mydim=myd, J = myJ*2, seed=123)
q15<-quantile(T_null_15,0.95)
hist(T_null_15, breaks = 30, main = "Monte Carlo Null Distribution of T_n", xlab = "T_n")


W_null_15<-simulate_W_null(n_mu, mydim=myd, J = myJ*2, seed=123)
qW15<-quantile(W_null_15,0.95)
hist(W_null_15, breaks = 30, main = "Monte Carlo Null Distribution of W_n", xlab = "T_n")






W<-6 #number of alternatives
incr<-seq(1,W,by=1)
mean1<-seq(1,W,by=1)
perc15<-rep(0, W)
perc15W<-rep(0, W)

#d=15
perc15<-rep(0, W)
for (w in 1:W){
  mylambda=mylambda
  epsilon=epsilon
  Q = 100
  set.seed=123
  n_nu <- n_mu * 5
  mean1[w]<- 0 + w*incr[w]/20 
  mean_ref<-0 
  mean_mu = rep(mean1[w],myd)
  mean_nu = rep(mean_ref,myd)
  mySigma<- diag(myd)
  Tn_out = NULL
  # Generate the point clouds
  for (q in 1:Q) {
    mu_points <- rmvt(n_mu, sigma = mySigma, df = t_df, delta = mean_mu)
    nu_points <- rmvt(n_nu, sigma = mySigma, df = t_df, delta = mean_nu)
    cost_matrix <- rdist(mu_points, nu_points)  
    a <- rep(1 / n_mu, n_mu)
    b <- rep(1 / n_nu, n_nu)
    Tn <- truncated_sinkhorn_loss(a, b, cost_matrix, mu_points, nu_points, epsilon, mylambda)
    if (Tn >= q15) Tn_out[q] = 1
    else Tn_out[q] =0
  }
  perc15[w]= sum(Tn_out)/Q
  print(w)
}

perc15W<-rep(0, W)
for (w in 1:W){
  mylambda=mylambda
  Q = 100
  set.seed=123
  n_nu <- n_mu * 5
  mean1[w]<- 0 + w*incr[w]/20
  mean_ref<-0 
  mean_mu = rep(mean1[w],myd)
  mean_nu = rep(mean_ref,myd)
  mySigma<- diag(myd)
  Tn_out = NULL
  # Generate the point clouds
  for (q in 1:Q) {
    mu_points <- rmvt(n_mu, sigma = mySigma, df = t_df, delta = mean_mu)
    nu_points <- rmvt(n_nu, sigma = mySigma, df = t_df, delta = mean_nu)
    cost_matrix <- rdist(mu_points, nu_points)  
    a <- rep(1 / n_mu, n_mu)
    b <- rep(1 / n_nu, n_nu)
    Tn <- transport::wasserstein(a,b,costm=cost_matrix, p=1)
    if (Tn >= qW15) Tn_out[q] = 1
    else Tn_out[q] =0
  }
  perc15W[w]= sum(Tn_out)/Q
  print(w)
}

xax<-mean1
plot(xax,perc15,type = "b", lwd=2, col="blue", main= "Power for d=25", xlab="Location shift", 
     ylab="Prob", ylim = c(0, 1))
lines(xax, perc15W,  type = "b", lwd = 2, col = "red",  lty = 3, pch = 15)
grid(lty = 3, col = "gray")

