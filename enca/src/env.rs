use crate::{
    config::Config,
    constants::{RO_CH_RNG, RW_CH_RNG},
    dataset::TrainExample,
    executors::NCAEnsembleExecutor,
    grid::Grid,
    nca::NCAEnsemble,
    substrate::Substrate,
    utils::mean,
};
use itertools::Itertools;
use ndarray::s;

pub fn compute_fitness(example: &TrainExample, executor: &NCAEnsembleExecutor, config: &Config) -> f32 {
    let mut executor = executor.clone();
    executor.run();

    let ensemble = &executor.ensemble;

    // The RW channels contain the output of the executor
    let pred_vis_slice = s![.., .., RW_CH_RNG];
    // The RO channels of the tgt substrate contain the ground truth.
    let tgt_vis_slice = s![.., .., RO_CH_RNG];

    let pred_substrate = &executor.executors.last().unwrap().substrate;
    let pred_vis_chs = pred_substrate.data.slice(pred_vis_slice);

    let mut tgt_grid = example.output.clone();
    ensemble.transform_pipeline.apply(&mut tgt_grid);
    let tgt_substrate = Substrate::from_grid(&tgt_grid);
    let out_vis_chs = tgt_substrate.data.slice(tgt_vis_slice);

    let diff = &pred_vis_chs - &out_vis_chs;
    let err = diff.pow2().mean().unwrap();

    let mut oscillation_cost = 0.0;
    let mut non_convergence_cost = 0.0;
    let mut l2_weight_cost = 0.0;
    let mut l1_weight_cost = 0.0;

    for nca_executor in &executor.executors {
        // Penalize for instability
        oscillation_cost += (&nca_executor.prev_substrate.data - &nca_executor.substrate.data)
            .pow2()
            .mean()
            .unwrap();

        non_convergence_cost += if nca_executor.steps == nca_executor.nca.max_steps {
            1.0
        } else {
            0.0
        };

        l2_weight_cost += mean(&nca_executor.nca.weights.iter().map(|w| w * w).collect_vec());
        l1_weight_cost += mean(&nca_executor.nca.weights.iter().map(|w| w.abs()).collect_vec());
    }

    err + config.oscillation_cost_coeff * oscillation_cost
        + config.non_convergence_cost_coeff * non_convergence_cost
        + config.l2_coeff * l2_weight_cost
        + config.l1_coeff * l1_weight_cost
}

#[inline]
pub fn inference(input: &Grid, ensemble: &NCAEnsemble) -> Grid {
    let mut executor = NCAEnsembleExecutor::new(ensemble.clone(), input);

    executor.run();

    let substrate = &executor.executors.last().unwrap().substrate;
    let mut pred_grid = substrate.to_grid();

    ensemble.transform_pipeline.revert(&mut pred_grid);

    pred_grid
}

pub fn eval(input: &Grid, output: &Grid, ensemble: &NCAEnsemble) -> f32 {
    let pred_grid = inference(input, ensemble);
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
