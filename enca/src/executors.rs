use std::fmt::Display;

use crate::{
    constants::HID_CH_RNG,
    grid::Grid,
    nca::{NCA, NCAEnsemble},
    substrate::Substrate,
};
use ndarray::s;
use ndarray_stats::QuantileExt;

/// Handles NCA step updates and stores execution state
#[derive(Clone)]
pub struct NCAExecutor {
    /// The NCA for this executor
    pub nca: NCA,
    /// Current state of the NCA substrate
    pub substrate: Substrate,
    /// Number of update steps completed so far
    pub steps: usize,
    /// Substrate state of the previous update
    pub prev_substrate: Substrate,
    /// Termination reason
    pub reason: Option<TerminationReason>,
}

impl NCAExecutor {
    pub fn new(nca: NCA, substrate: Substrate) -> Self {
        let prev = substrate.clone();
        Self {
            nca,
            substrate,
            steps: 0,
            prev_substrate: prev,
            reason: None,
        }
    }

    /// Runs until termination criteria is met.
    pub fn run(&mut self) -> TerminationReason {
        loop {
            if let Some(reason) = self.step() {
                break reason;
            }
        }
    }

    /// Executes one iteration step. Returns `Some` if max_steps reached or convergence.
    pub fn step(&mut self) -> Option<TerminationReason> {
        // Stop when max steps reached
        if self.steps >= self.nca.max_steps {
            self.reason = Some(TerminationReason::MaxSteps);
            return self.reason.clone();
        }

        self.prev_substrate = self.substrate.clone();

        self.nca.update(&mut self.substrate);
        self.steps += 1;

        // Stop on convergence
        // if (&self.prev_substrate.data - &self.substrate.data).abs().mean().unwrap() < 1e-5 {
        if *(&self.prev_substrate.data - &self.substrate.data).abs().max().unwrap() < 0.25 {
            self.reason = Some(TerminationReason::Convergence { steps: self.steps });
            return self.reason.clone();
        }

        None
    }
}

/// Ensemble of NCAs that collectively solves a task
#[derive(Clone)]
pub struct NCAEnsembleExecutor {
    pub ensemble: NCAEnsemble,
    /// Individual executors for each NCA
    pub executors: Vec<NCAExecutor>,
    /// Index of active executor
    pub curr_exec_idx: usize,
    /// Current iteration steps
    pub steps: usize,
    /// Termination reasons for individual executors
    pub reasons: Vec<Option<TerminationReason>>,
}

impl NCAEnsembleExecutor {
    pub fn new(ensemble: NCAEnsemble, grid: &Grid) -> Self {
        let mut grid = grid.clone();
        ensemble.transform_pipeline.apply(&mut grid);

        let substrate = Substrate::from_grid(&grid);

        let executors = ensemble
            .ncas
            .iter()
            .map(|nca| NCAExecutor::new(nca.clone(), substrate.clone()))
            .collect();
        let n_ncas = ensemble.ncas.len();

        Self {
            ensemble,
            executors,
            curr_exec_idx: 0,
            steps: 0,
            reasons: vec![None; n_ncas],
        }
    }

    pub fn upsert_nca(&mut self, nca: NCA, i: usize) {
        let mut new_executor = self.executors.last().unwrap().clone();
        let mut substrate = new_executor.substrate.clone();

        // Clear hidden state
        substrate.data.slice_mut(s![.., .., HID_CH_RNG]).fill(0.0);

        // Copy substrate
        new_executor.substrate = substrate.clone();
        new_executor.prev_substrate = substrate;
        new_executor.steps = 0;
        new_executor.nca = nca.clone();
        new_executor.reason = None;
        self.reasons.push(None);

        if i >= self.executors.len() {
            self.executors.push(new_executor);
            self.ensemble.ncas.push(nca);
            self.curr_exec_idx += 1;
        } else {
            self.executors[i] = new_executor;
            self.ensemble.ncas[i] = nca;
        }
    }

    /// Sequentially executes NCAs until termination criteria is met
    pub fn run(&mut self) -> TerminationReason {
        loop {
            if let Some(reason) = self.step() {
                break reason;
            }
        }
    }

    /// Executes one iteration step
    pub fn step(&mut self) -> Option<TerminationReason> {
        let cur_executor = self.executors.get_mut(self.curr_exec_idx).unwrap();

        let reason = cur_executor.step();
        self.reasons[self.curr_exec_idx] = reason.clone();
        self.steps += 1;

        // Continue with current executor
        reason.as_ref()?;

        // Move to next executor
        if self.curr_exec_idx < self.executors.len() - 1 {
            let mut substrate = self.executors[self.curr_exec_idx].substrate.clone();

            // Clear hidden state
            substrate.data.slice_mut(s![.., .., HID_CH_RNG]).fill(0.0);

            self.curr_exec_idx += 1;

            // Copy previous substrate state
            self.executors[self.curr_exec_idx].substrate = substrate.clone();
            self.executors[self.curr_exec_idx].prev_substrate = substrate;

            return None;
        }

        reason
    }
}

#[derive(Clone, PartialEq, Debug)]
pub enum TerminationReason {
    MaxSteps,
    Convergence { steps: usize },
}

impl Display for TerminationReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TerminationReason::MaxSteps => write!(f, "MaxSteps"),
            TerminationReason::Convergence { steps } => write!(f, "Convergence: {steps}"),
        }
    }
}
