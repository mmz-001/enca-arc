use indexmap::IndexMap;
use itertools::Itertools;

use crate::{env::inference, grid::Grid, nca::NCAEnsemble};

pub fn vote(grid: &Grid, ensembles: &Vec<NCAEnsemble>, k: usize, verbose: bool) -> Vec<NCAEnsemble> {
    let mut pred_counts = IndexMap::<u64, (NCAEnsemble, usize)>::new();

    for ensemble in ensembles {
        let pred_grid = inference(grid, ensemble);
        let hash = pred_grid.get_hash();
        if hash == grid.get_hash() {
            continue;
        }
        pred_counts
            .entry(hash)
            .and_modify(|(_, count)| *count += 1)
            .or_insert((ensemble.clone(), 1));
    }

    // Collect and sort by count descending, stable to preserve deterministic order
    // for equal counts based on insertion order.
    let mut entries: Vec<(u64, (NCAEnsemble, usize))> = pred_counts.into_iter().collect();

    if entries.is_empty() {
        return ensembles.clone().into_iter().take(k).collect_vec();
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

    entries
        .into_iter()
        .take(k)
        .map(|(_, (ensemble, _count))| ensemble)
        .collect()
}
