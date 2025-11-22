use crate::{
    executors::{cpu::NCAExecutorCpu, gpu::NCAExecutorGpu},
    grid::Grid,
    nca::NCA,
    substrate::Substrate,
};
use serde::{Deserialize, Serialize};
pub mod cpu;
pub mod gpu;

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub enum Backend {
    CPU,
    GPU,
}

enum NCAExecutorInner {
    Cpu(NCAExecutorCpu),
    Gpu(NCAExecutorGpu),
}

pub struct NCAExecutor {
    inner: NCAExecutorInner,
}

impl NCAExecutor {
    pub fn new(nca: NCA, grid: &Grid, backend: Backend) -> Self {
        match backend {
            Backend::CPU => Self {
                inner: NCAExecutorInner::Cpu(NCAExecutorCpu::new(nca, grid)),
            },
            Backend::GPU => Self {
                inner: NCAExecutorInner::Gpu(NCAExecutorGpu::new(nca, grid)),
            },
        }
    }

    pub fn run(&mut self) {
        match &mut self.inner {
            NCAExecutorInner::Cpu(cpu) => cpu.run(),
            NCAExecutorInner::Gpu(gpu) => gpu.run(),
        }
    }

    pub fn substrate(&self) -> &Substrate {
        match &self.inner {
            NCAExecutorInner::Cpu(cpu) => &cpu.substrate,
            NCAExecutorInner::Gpu(gpu) => gpu.substrate(),
        }
    }

    pub fn step(&mut self) -> bool {
        match &mut self.inner {
            NCAExecutorInner::Cpu(cpu) => cpu.step(),
            NCAExecutorInner::Gpu(_) => panic!("step() not implemented for GPU backend"),
        }
    }

    pub fn sup_steps(&self) -> usize {
        match &self.inner {
            NCAExecutorInner::Cpu(cpu) => cpu.sup_steps,
            NCAExecutorInner::Gpu(_) => panic!("sup_steps not implemented for GPU backend"),
        }
    }

    pub fn rec_steps(&self) -> usize {
        match &self.inner {
            NCAExecutorInner::Cpu(cpu) => cpu.rec_steps,
            NCAExecutorInner::Gpu(_) => panic!("rec_steps not implemented for GPU backend"),
        }
    }

    pub fn hid_steps(&self) -> usize {
        match &self.inner {
            NCAExecutorInner::Cpu(cpu) => cpu.hid_steps,
            NCAExecutorInner::Gpu(_) => panic!("hid_steps not implemented for GPU backend"),
        }
    }

    pub fn nca(&self) -> &NCA {
        match &self.inner {
            NCAExecutorInner::Cpu(cpu) => &cpu.nca,
            NCAExecutorInner::Gpu(gpu) => gpu.nca(),
        }
    }
}
