use crate::{grid::Grid, nca::NCA, substrate::Substrate};

/// Handles NCA step updates and stores execution state
#[derive(Clone)]
pub struct NCAExecutorCpu {
    pub nca: NCA,
    pub steps: usize,
    pub substrate: Substrate,
}

impl NCAExecutorCpu {
    pub fn new(nca: NCA, grid: &Grid) -> Self {
        let mut grid = grid.clone();

        nca.transform_pipeline.apply(&mut grid);
        let substrate = Substrate::from_grid(&grid);

        Self {
            nca,
            steps: 0,
            substrate,
        }
    }

    pub fn run(&mut self) {
        for _ in 0..self.nca.max_steps {
            self.step();
        }
    }

    /// Executes one iteration step.
    pub fn step(&mut self) -> bool {
        if self.steps >= self.nca.max_steps {
            return true;
        }
        self.nca.update(&mut self.substrate);
        self.steps += 1;
        false
    }
}
