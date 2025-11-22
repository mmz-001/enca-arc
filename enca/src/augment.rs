use std::collections::HashSet;

use indexmap::IndexMap;
use itertools::Itertools;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    config::Config,
    constants::MAX_PERMUTATIONS,
    dataset::Task,
    executors::NCAExecutor,
    grid::Grid,
    nca::NCA,
    transforms::{RemapColors, Transform},
    utils::union_sets,
};

#[inline]
fn colors_sorted_nonzero(set: &HashSet<u8>) -> Vec<u8> {
    set.iter().copied().filter(|&c| c != 0).sorted().collect()
}

pub fn augment(grid: &Grid, task: &Task, nca: NCA, config: &Config, rng: &mut impl Rng) -> NCA {
    let ti_col_set = union_sets(task.train_inputs().iter().map(|grid| grid.colors().clone()));
    let grid_col_set = grid.colors();

    if grid_col_set.difference(&ti_col_set).next().is_none() {
        // No unseen colors. Return original
        return nca;
    }

    let ti_cols = colors_sorted_nonzero(&ti_col_set);
    let grid_cols = colors_sorted_nonzero(grid_col_set);

    if grid_cols.len() > ti_cols.len() {
        // Grid has more colors than all input colors. Can't remap; return original
        return nca;
    }

    let n = ti_cols.len();
    let k = grid_cols.len();
    let total = n_pk(n, k);
    let mut pred_grid_counts = IndexMap::<u64, (usize, RemapColors)>::new();
    let empty_grid_hash = Grid::from_vec(vec![vec![0; grid.width()]; grid.height()]).get_hash();

    for rank in floyd_unique_indices(total, MAX_PERMUTATIONS.min(config.max_fun_evals), rng) {
        // Sample up to MAX_PERMUTATIONS unique color remappings and get majority vote
        let perm = unrank_k_perm(rank, &ti_cols, k);
        let mut color_transform = RemapColors::new();

        for (grid_col, map_col) in grid_cols.iter().zip(perm.iter()) {
            color_transform.map(*grid_col, *map_col);
        }

        let mut aug_nca = nca.clone();
        aug_nca
            .transform_pipeline
            .steps
            .insert(0, Transform::RemapColors(color_transform.clone()));

        let mut executor = NCAExecutor::new(aug_nca.clone(), grid, config.backend.clone());

        executor.run();

        let mut pred_grid = executor.substrate().to_grid();
        aug_nca.transform_pipeline.revert(&mut pred_grid);

        // Don't count empty grids:
        if pred_grid.get_hash() == empty_grid_hash {
            continue;
        }

        pred_grid_counts
            .entry(pred_grid.get_hash())
            .and_modify(|(count, _)| *count += 1)
            .or_insert((1, color_transform.clone()));
    }

    if pred_grid_counts.is_empty() {
        return nca;
    }

    let pred_grid_counts_sorted = pred_grid_counts
        .clone()
        .into_iter()
        .sorted_by_key(|(_, (count, _))| *count)
        .rev()
        .collect_vec();

    let maj_transform = pred_grid_counts_sorted[0].1.1.clone();

    let mut aug_nca = nca.clone();
    aug_nca
        .transform_pipeline
        .steps
        .insert(0, Transform::RemapColors(maj_transform));

    aug_nca
}

/// The trained NCA and augmented NCAs for each test problem.
#[derive(Serialize, Deserialize, Clone)]
pub struct TaskNCAs {
    pub train: NCA,
    pub test: Vec<NCA>,
}

/// Compute nPk = n * (n-1) * ... * (n-k+1) as u128
fn n_pk(n: usize, k: usize) -> u128 {
    let mut acc = 1u128;
    for i in 0..k {
        acc = acc.saturating_mul((n - i) as u128);
    }
    acc
}

// Floyd's algorithm to sample m unique indices from [0, n_total) without replacement
fn floyd_unique_indices<R: rand::Rng>(n_total: u128, m: usize, rng: &mut R) -> Vec<u128> {
    let m = m.min(n_total as usize);
    let mut chosen = HashSet::<u128>::with_capacity(m);
    let mut out = Vec::with_capacity(m);
    let start = n_total - m as u128;
    for j in start..n_total {
        let t = rng.random_range(0..=j);
        let x = if chosen.contains(&t) { j } else { t };
        chosen.insert(x);
        out.push(x);
    }
    out
}

// Unrank a k-permutation (arrangement) of items without replacement using mixed radix
fn unrank_k_perm(rank: u128, items: &[u8], k: usize) -> Vec<u8> {
    let n = items.len();
    let mut r = rank;
    let mut avail: Vec<u8> = items.to_vec();
    let mut out = Vec::with_capacity(k);
    for i in 0..k {
        let base = (n - i) as u128;
        let idx = (r % base) as usize;
        r /= base;
        out.push(avail.remove(idx));
    }
    out
}
