use serde::{Deserialize, Serialize};

use crate::executors::Backend;

/// Hyperparameters for the ENCA algorithm
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Config {
    /// Number of epochs for the evolutionary loop
    pub epochs: usize,
    pub sup_steps: usize,
    pub rec_steps: usize,
    pub hid_steps: usize,
    /// Population size for evolutionary loop
    pub pop: usize,
    /// Tournament selection size
    pub k: usize,
    /// Only a subset of the parameters is used to for CMA-ES
    /// optimization
    pub subset_size: usize,
    /// Maximum number of function evaluations per CMA-ES run
    pub max_fun_evals: usize,
    /// L2 weight decay coefficient
    pub l2_coeff: f64,
    /// Inference backend; GPU or CPU
    pub backend: Backend,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            epochs: 50,
            sup_steps: 8,
            rec_steps: 3,
            hid_steps: 6,
            pop: 12,
            k: 2,
            subset_size: 488,
            max_fun_evals: 500,
            l2_coeff: 5e-5,
            backend: Backend::GPU,
        }
    }
}
