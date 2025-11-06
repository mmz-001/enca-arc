use crate::constants::N_PARAMS;
use crate::{constants::INP_CHS, grid::Grid, nca::NCA, substrate::Substrate};
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use itertools::Itertools;

use std::env;
use std::sync::{Arc, LazyLock};

#[derive(Clone)]
pub struct NCAExecutorGpu {
    inner: NCAExecutorGpuBatch,
}

impl NCAExecutorGpu {
    pub fn new(nca: NCA, grid: &Grid) -> Self {
        Self {
            inner: NCAExecutorGpuBatch::new(nca, std::slice::from_ref(&grid)),
        }
    }

    /// Execute NCA
    pub fn run(&mut self) {
        self.inner.run();
    }

    pub fn substrate(&self) -> &Substrate {
        &self.inner.inner.individuals[0].substrates[0]
    }

    pub fn nca(&self) -> &NCA {
        &self.inner.inner.individuals[0].nca
    }
}

#[derive(Clone)]
pub struct NCAExecutorGpuBatch {
    inner: PopNCAExecutorGpuBatch,
}

impl NCAExecutorGpuBatch {
    pub fn new(nca: NCA, grids: &[&Grid]) -> Self {
        Self {
            inner: PopNCAExecutorGpuBatch::new(vec![nca], grids),
        }
    }

    pub fn run(&mut self) {
        self.inner.run();
    }

    pub fn substrates(&self) -> &Vec<Substrate> {
        &self.inner.individuals[0].substrates
    }
}

#[derive(Clone)]
pub struct Individual {
    pub nca: NCA,
    pub substrates: Vec<Substrate>,
}

#[derive(Clone)]
pub struct PopNCAExecutorGpuBatch {
    pub individuals: Vec<Individual>,
}

impl PopNCAExecutorGpuBatch {
    pub fn new(ncas: Vec<NCA>, grids: &[&Grid]) -> Self {
        let individuals = ncas
            .into_iter()
            .map(|nca| {
                let substrates = grids
                    .iter()
                    .map(|grid| {
                        let mut grid = (*grid).clone();
                        nca.transform_pipeline.apply(&mut grid);
                        Substrate::from_grid(&grid)
                    })
                    .collect_vec();
                Individual { nca, substrates }
            })
            .collect();

        Self { individuals }
    }

    pub fn run(&mut self) {
        let substrates_0 = &self.individuals[0].substrates;

        let widths = substrates_0
            .iter()
            .map(|substrate| substrate.width as i32)
            .collect_vec();
        let heights = substrates_0
            .iter()
            .map(|substrate| substrate.height as i32)
            .collect_vec();

        let max_grid_size = widths.iter().zip(&heights).map(|(w, h)| w * h).max().unwrap();

        if max_grid_size > 1024 {
            panic!("Grids with more than 1024 elements not supported.")
        }

        let max_steps_all_equal = self.individuals.iter().map(|ind| ind.nca.max_steps).all_equal();

        if !max_steps_all_equal {
            panic!("Every individual in the population should have equal max_steps")
        }

        let pop_size = self.individuals.len();
        let sub_max_len = INP_CHS * max_grid_size as usize;
        let ind_subs_total_len = sub_max_len * substrates_0.len();
        let pop_sub_total_len = ind_subs_total_len * pop_size;
        let mut pop_substrates = vec![0.0; pop_sub_total_len];

        for (ind_idx, ind) in self.individuals.iter().enumerate() {
            for (i, s) in ind.substrates.iter().enumerate() {
                let start = ind_idx * ind_subs_total_len + i * sub_max_len;
                let dst = &mut pop_substrates[start..start + s.data.len()];
                dst.copy_from_slice(s.data.as_slice().unwrap());
            }
        }

        let mut pop_nca_params = vec![0.0; pop_size * N_PARAMS];

        for (ind_idx, ind) in self.individuals.iter().enumerate() {
            let nca = &ind.nca;
            let nca_params = [nca.weights.clone(), nca.biases.clone()].concat();
            let start = ind_idx * N_PARAMS;
            let dst = &mut pop_nca_params[start..start + N_PARAMS];
            dst.copy_from_slice(&nca_params);
        }

        let ctxs = &*CUDA;
        // TODO: figure out a better way to distribute work
        let (_, kernel, stream) = &ctxs[rayon::current_thread_index().unwrap_or(0) % ctxs.len()];

        let mut d_pop_subs = stream.memcpy_stod(&pop_substrates).unwrap();
        let d_pop_nca_params = stream.memcpy_stod(&pop_nca_params).unwrap();
        let d_heights = stream.memcpy_stod(&heights).unwrap();
        let d_widths = stream.memcpy_stod(&widths).unwrap();
        let max_steps = self.individuals[0].nca.max_steps as i32;
        let n_grids = substrates_0.len() as i32;
        let mut builder = stream.launch_builder(kernel);

        builder.arg(&mut d_pop_subs);
        builder.arg(&d_pop_nca_params);
        builder.arg(&d_heights);
        builder.arg(&d_widths);
        builder.arg(&max_steps);
        builder.arg(&max_grid_size);

        let lc = LaunchConfig {
            grid_dim: (n_grids as u32, pop_size as u32, 1),
            block_dim: (max_grid_size as u32, 1, 1),
            shared_mem_bytes: (max_grid_size as usize * INP_CHS * core::mem::size_of::<f32>()) as u32,
        };

        unsafe { builder.launch(lc) }.unwrap();

        let pop_substrates = stream.memcpy_dtov(&d_pop_subs).unwrap();

        for (ind_idx, ind) in self.individuals.iter_mut().enumerate() {
            for i in 0..ind.substrates.len() {
                let start = ind_idx * ind_subs_total_len + i * sub_max_len;
                let sub_slice = &pop_substrates[start..start + ind.substrates[i].data.len()];
                ind.substrates[i]
                    .data
                    .as_slice_mut()
                    .unwrap()
                    .copy_from_slice(sub_slice);
            }
        }
    }
}

type T = Vec<(Arc<CudaContext>, Arc<CudaFunction>, Arc<CudaStream>)>;

pub static CUDA: LazyLock<T> = LazyLock::new(|| {
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        include_str!("./kernels.cu"),
        cudarc::nvrtc::CompileOptions {
            fmad: Some(true),
            include_paths: vec![concat!(env!("CARGO_MANIFEST_DIR"), "/src/executors/gpu").to_string()],
            ..Default::default()
        },
    )
    .unwrap();

    let device_count = cudarc::runtime::result::device::get_count().unwrap() as usize;
    // let device_count = 1;
    let n_threads = rayon::current_num_threads();
    println!("======Initializing GPU(s)=========");
    println!("GPU count={}", device_count);
    println!("Rayon thread count={}", n_threads);

    let threads_per_device = n_threads.div_ceil(device_count);
    // let threads_per_device = 1;
    let mut ctxs = Vec::with_capacity(threads_per_device * device_count);

    for dev_ord in 0..device_count {
        let ctx = cudarc::driver::CudaContext::new(dev_ord).unwrap();
        let module = ctx.load_module(ptx.clone()).unwrap();
        let kernel = Arc::new(module.load_function("pop_nca_executor_run_batch").unwrap());

        ctxs.extend((0..threads_per_device).map(|_| (ctx.clone(), kernel.clone(), ctx.new_stream().unwrap())))
    }
    println!("======GPU(s) Ready================\n");

    ctxs
});
