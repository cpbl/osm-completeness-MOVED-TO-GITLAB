
data {
    int<lower=0> N;                     // number of observations
    int<lower=0> J;                     // number of countries
    int<lower=0> L;                     // number of grid-cell level covariates (no intercept because no omitted dummyies)
    int<lower=0> M;                     // number of country-level covariates (excl. intercept)
    int<lower=0> obsSegs[N];            // observed segments
    int<lower=0> totSegs[N];            // total segments
    int<lower=0> country[N];            // country id of each observation
    real logdensity[N];                 // density of each observation
    int<lower=1,upper=5> quintile[N];   // categorical variable for density quintile of each observation
    real<lower=0> wght[N];              // inverse probability weights of each observation (used only to aggregate sample to country level)
    
    // Data for out-of-sample prediction. Let's not do this...
    //int<lower=0> N_oos;
    //int<lower=0> country_oos[N_oos];
    //real logdensity_oos[N_oos];
    
    // country-level covariates
    row_vector[J] Z1;
    row_vector[J] Z2;
    row_vector[J] Z3;
    row_vector[J] Z4;       
}

transformed data {
    // Convert country-level predictors to a matrix and add intercept
    // Actually, it's a row vector of lists -- I can't get the matrix assignment to work

    row_vector[M+1] Z[J];
    for (j in 1:J) {
        Z[j][1] <- 1.0;
        Z[j][2] <- Z1[j];
        Z[j][3] <- Z2[j];
        Z[j][4] <- Z3[j];
        Z[j][5] <- Z4[j];
    } 
    
}

parameters {

    // This section sets up the covariance matrix. See Stan manual 6.12 for approach (this is basically identical)
    // Separate covariance matrices for Poisson and Binomial models (since by design they are independent)
    corr_matrix[(L)] Omega_psn;               // decomposed correlation matrix for individual-level predictors. 
    corr_matrix[(L)] Omega_bin;                  
    vector<lower=0>[L] tau_psn;           // scale of diagonals
    vector<lower=0>[L] tau_bin;             
    matrix[M+1,L] gamma_psn;                // country-level coefficients (including the intercept)
    matrix[M+1,L] gamma_bin;
    vector[L] beta_psn[J];                  // grid-cell level coefficients (including the intercept). 
    vector[L] beta_bin[J];                  // grid-cell level coefficients (including the intercept). 
}

transformed parameters {
        
	vector[N] lambda;               // log of Poisson mean  
	vector[N] theta;                // logit of binomial probability             
	row_vector[L] z_gamma_psn[J];	
	row_vector[L] z_gamma_bin[J];	
	for (j in 1:J) {
	    z_gamma_psn[j] <- Z[j] * gamma_psn;
	    z_gamma_bin[j] <- Z[j] * gamma_bin;
	}
	
	
	for (i in 1:N){
        lambda[i] <- beta_psn[country[i]][quintile[i]] + logdensity[i] * beta_psn[country[i]][6];
        theta[i]  <- beta_bin[country[i]][quintile[i]] + logdensity[i] * beta_bin[country[i]][6];
    }
}

model {
  	// Weakly informative priors for covariance matrix and second-level parameters
    tau_psn ~ cauchy(0,2.5);
    tau_bin ~ cauchy(0,2.5);
    Omega_psn ~ lkj_corr(2.0);
    Omega_bin ~ lkj_corr(2.0);
  	to_vector(gamma_psn) ~ cauchy(0, 5);
  	to_vector(gamma_bin) ~ cauchy(0, 5);

  	// Model
  	beta_psn ~ multi_normal(z_gamma_psn, quad_form_diag(Omega_psn, tau_psn));
  	beta_bin ~ multi_normal(z_gamma_bin, quad_form_diag(Omega_bin, tau_bin));
	totSegs ~ poisson_log(lambda);
	obsSegs ~ binomial_logit(totSegs, theta);
}

generated quantities {      
    // separate out matrices into columns to aid plotting
    vector[J] b1_psn;
    vector[J] b2_psn;
    vector[J] b3_psn;
    vector[J] b4_psn;
    vector[J] b5_psn;
    vector[J] b6_psn;
    vector[J] b1_bin;
    vector[J] b2_bin;
    vector[J] b3_bin;
    vector[J] b4_bin;
    vector[J] b5_bin;
    vector[J] b6_bin;
    row_vector[M+1] gamma1_psn;
    row_vector[M+1] gamma2_psn;
    row_vector[M+1] gamma3_psn;
    row_vector[M+1] gamma4_psn;
    row_vector[M+1] gamma5_psn;
    row_vector[M+1] gamma6_psn;
    row_vector[M+1] gamma1_bin;
    row_vector[M+1] gamma2_bin;    
    row_vector[M+1] gamma3_bin;    
    row_vector[M+1] gamma4_bin;    
    row_vector[M+1] gamma5_bin;    
    row_vector[M+1] gamma6_bin;    
    
    // calculate predictive accuracy in terms of what we care about
    real totSegs_hat[N];        // predicted value
    real totSegs_hat_agg[J];    // predicted, aggregated by country
    real totSegs_agg[J];        // actual,    aggregated by country
    real osmSegs_hat_agg[J];
    real predError_pc[J];       // country level prediction error for totSegs
    real fc_hat[J];             // predicted fraction complete by country
    real fc_hat_num[J];  
    real fc_hat_denom[J]; 
    real fc[J];                 // actual fraction complete by country
    real fc_num[J];  
    real fc_denom[J]; 
    
    for (j in 1:J){ 
        // pull out vector of rows of beta, makes it possible to use pystan plotting function
        b1_psn[j] <- beta_psn[j][1];
        b2_psn[j] <- beta_psn[j][2];
        b3_psn[j] <- beta_psn[j][3];
        b4_psn[j] <- beta_psn[j][4];
        b5_psn[j] <- beta_psn[j][5];
        b6_psn[j] <- beta_psn[j][6];
        b1_bin[j] <- beta_bin[j][1];
        b2_bin[j] <- beta_bin[j][2];
        b3_bin[j] <- beta_bin[j][3];
        b4_bin[j] <- beta_bin[j][4];
        b5_bin[j] <- beta_bin[j][5];
        b6_bin[j] <- beta_bin[j][6];
    }
    gamma1_psn <- gamma_psn'[1];
    gamma2_psn <- gamma_psn'[2];
    gamma3_psn <- gamma_psn'[3];
    gamma4_psn <- gamma_psn'[4];
    gamma5_psn <- gamma_psn'[5];
    gamma6_psn <- gamma_psn'[6];
    gamma1_bin <- gamma_bin'[1];
    gamma2_bin <- gamma_bin'[2];
    gamma3_bin <- gamma_bin'[3];
    gamma4_bin <- gamma_bin'[4];
    gamma5_bin <- gamma_bin'[5];
    gamma6_bin <- gamma_bin'[6];

    for (j in 1:J){                 // Initialize country-level totals
        totSegs_agg[j] <- 0;
        totSegs_hat_agg[j] <- 0;
        osmSegs_hat_agg[j] <- 0;
        fc_num[j]       <- 0;
        fc_hat_num[j]   <- 0;
        fc_denom[j]     <- 0;        
        fc_hat_denom[j] <- 0;       
    }

    for(i in 1:N){
        totSegs_hat[i] <- exp(lambda[i]); 
        totSegs_hat_agg[country[i]] <- totSegs_hat_agg[country[i]] + totSegs_hat[i];
        totSegs_agg[country[i]] <- totSegs_agg[country[i]] + totSegs[i];
        osmSegs_hat_agg[country[i]] <- osmSegs_hat_agg[country[i]] + totSegs_hat[i] * 1.0 * inv_logit(theta[i]);
        
        // numerator and denominator are weighted
        fc_num[country[i]]       <- fc_num[country[i]]       + obsSegs[i]     * 1.0 / wght[i];
        fc_hat_num[country[i]]   <- fc_hat_num[country[i]]   + totSegs_hat[i] * 1.0 * inv_logit(theta[i]) / wght[i];
        fc_denom[country[i]]     <- fc_denom[country[i]]     + totSegs[i]     * 1.0 / wght[i];
        fc_hat_denom[country[i]] <- fc_hat_denom[country[i]] + totSegs_hat[i] * 1.0 / wght[i];
    }
             
    for(j in 1:J){      // aggregate country-level data
        predError_pc[j] <- totSegs_hat_agg[j] * 1.0 / totSegs_agg[j];
        fc[j]         <- fc_num[j]     * 1.0 / fc_denom[j];
        fc_hat[j]     <- fc_hat_num[j] * 1.0 / fc_hat_denom[j];
    }
    

}
