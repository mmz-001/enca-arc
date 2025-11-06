use serde::{Deserialize, Serialize};

use crate::executors::Backend;

/// Hyperparameters for the ENCA algorithm
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Config {
    /// Number of epochs for the evolutionary loop
    pub epochs: usize,
    /// Maximum number of NCA steps
    pub max_steps: usize,
    /// Population size for evolutionary loop
    pub pop: usize,
    /// Tournament selection size
    pub k: usize,
    /// Only a subset of the parameters is used to for CMA-ES
    /// optimization
    pub subset_size: usize,
    /// Maximum number of function evaluations per CMA-ES run
    pub max_fun_evals: usize,
    /// CMA-ES initial sigma
    pub initial_sigma: f64,
    /// L2 weight decay coefficient
    pub l2_coeff: f64,
    /// Inference backend; GPU or CPU
    pub backend: Backend,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            epochs: 20,
            max_steps: 40,
            pop: 12,
            k: 2,
            subset_size: 120,
            max_fun_evals: 5000,
            initial_sigma: 0.2,
            l2_coeff: 1e-4,
            backend: Backend::GPU,
        }
    }
}
