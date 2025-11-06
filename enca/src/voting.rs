use indexmap::IndexMap;
use itertools::Itertools;

use crate::{env::inference, executors::Backend, grid::Grid, nca::NCA};

pub fn vote(grid: &Grid, ncas: &Vec<NCA>, k: usize, verbose: bool, backend: Backend) -> Vec<NCA> {
    let mut pred_counts = IndexMap::<u64, (NCA, usize)>::new();

    for nca in ncas {
        let pred_grid = inference(grid, nca, backend.clone());
        let hash = pred_grid.get_hash();
        if hash == grid.get_hash() {
            continue;
        }
        pred_counts
            .entry(hash)
            .and_modify(|(_, count)| *count += 1)
            .or_insert((nca.clone(), 1));
    }

    // Collect and sort by count descending, stable to preserve deterministic order
    // for equal counts based on insertion order.
    let mut entries: Vec<(u64, (NCA, usize))> = pred_counts.into_iter().collect();

    if entries.is_empty() {
        return ncas.clone().into_iter().take(k).collect_vec();
    }

    entries.sort_by(|a, b| b.1.1.cmp(&a.1.1));

    if verbose {
        let hash_counts = entries
            .iter()
            .map(|(hash, (_, count))| (hash, count))
            .collect::<Vec<_>>();

        println!("\nMajority vote results:");
        let mut ceq1 = 0;
        for (hash, count) in &hash_counts {
            if **count > 1 {
                println!("hash:{hash}, count:{count}")
            } else {
                ceq1 += 1
            }
        }
        if ceq1 > 0 {
            println!("{ceq1} grids with only one count")
        }
    }

    entries.into_iter().take(k).map(|(_, (nca, _count))| nca).collect()
}
