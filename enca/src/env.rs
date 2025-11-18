use crate::{
    config::Config,
    constants::{RO_CH_RNG, RW_CH_RNG},
    dataset::TrainExample,
    executors::{
        Backend, NCAExecutor,
        cpu::NCAExecutorCpu,
        gpu::{Individual, PopNCAExecutorGpuBatch},
    },
    grid::Grid,
    nca::NCA,
    substrate::Substrate,
    utils::mean,
};
use itertools::Itertools;
use ndarray::s;

pub fn compute_fitness_pop(examples: &[TrainExample], ncas: Vec<NCA>, config: &Config) -> Vec<f64> {
    let pop_size = ncas.len();
    let grids = examples.iter().map(|example| &example.input).collect_vec();

    let population = match config.backend {
        Backend::CPU => {
            let mut population = Vec::with_capacity(pop_size);
            for nca in ncas {
                let substrates = examples
                    .iter()
                    .map(|example| {
                        let mut executor = NCAExecutorCpu::new(nca.clone(), &example.input);
                        executor.run();
                        executor.substrate
                    })
                    .collect_vec();
                population.push(Individual { nca, substrates });
            }
            population
        }
        Backend::GPU => {
            let mut executor = PopNCAExecutorGpuBatch::new(ncas, &grids);
            executor.run();
            executor.individuals
        }
    };

    let mut fitnesses = Vec::with_capacity(pop_size);

    for individual in population {
        let mut fitness = 0.0f64;
        let pred_substrates = individual.substrates;
        let nca = individual.nca;

        for (example, pred_substrate) in examples.iter().zip(pred_substrates) {
            let pred_vis_chs = pred_substrate.data.slice(s![.., .., RW_CH_RNG]);

            let mut tgt_grid = example.output.clone();
            nca.transform_pipeline.apply(&mut tgt_grid);
            let tgt_substrate = Substrate::from_grid(&tgt_grid);
            let out_vis_chs = tgt_substrate.data.slice(s![.., .., RO_CH_RNG]);

            let diff = &pred_vis_chs - &out_vis_chs;
            let err = diff.mapv(f64::from).pow2().mean().unwrap();

            fitness += err
        }

        let l2_weight_cost = mean(&nca.weights.iter().map(|w| (*w as f64) * (*w as f64)).collect_vec());

        fitness += config.l2_coeff * l2_weight_cost;

        fitnesses.push(fitness / examples.len() as f64)
    }

    fitnesses
}

#[inline]
pub fn inference(input: &Grid, nca: &NCA, backend: Backend) -> Grid {
    let mut executor = NCAExecutor::new(nca.clone(), input, backend);

    executor.run();

    let mut pred_grid = executor.substrate().to_grid();

    nca.transform_pipeline.revert(&mut pred_grid);

    pred_grid
}

pub fn eval(input: &Grid, output: &Grid, nca: &NCA, backend: Backend) -> f32 {
    let pred_grid = inference(input, nca, backend);
    compute_accuracy(&pred_grid, output)
}

fn compute_accuracy(pred_grid: &Grid, target_grid: &Grid) -> f32 {
    if pred_grid.shape() != target_grid.shape() {
        return 0.0;
    }

    let height = pred_grid.height();
    let width = pred_grid.width();

    let mut correct: usize = 0;
    let total: usize = height * width;

    for yi in 0..height {
        for xi in 0..width {
            let gt_col = target_grid[(yi, xi)];
            let pred_col = pred_grid[(yi, xi)];

            if gt_col == pred_col {
                correct += 1;
            }
        }
    }

    correct as f32 / total as f32
}
