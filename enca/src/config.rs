use serde::{Deserialize, Serialize};

// Hyperparameters for the ENCA algorithm
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Config {
    pub max_ncas: usize,
    pub max_steps: usize,
    pub pop: usize,
    pub k: usize,
    pub cmaes_max_fun_evals: usize,
    pub cmaes_initial_sigma: f32,
    pub oscillation_cost_coeff: f32,
    pub non_convergence_cost_coeff: f32,
    pub l1_coeff: f32,
    pub l2_coeff: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_ncas: 5,
            max_steps: 40,
            pop: 100,
            k: 5,
            cmaes_max_fun_evals: 20_000,
            cmaes_initial_sigma: 0.2,
            oscillation_cost_coeff: 1e-5,
            non_convergence_cost_coeff: 1e-5,
            l1_coeff: 1e-4,
            l2_coeff: 1e-4,
        }
    }
}
