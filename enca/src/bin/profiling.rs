use enca::{config::Config, dataset::Dataset, executors::gpu::PopNCAExecutorGpuBatch, nca::NCA};
use itertools::Itertools;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn main() {
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let tasks_path = "./data/v2/arc-agi_evaluation_challenges.json";
    let dataset = Dataset::load(tasks_path, None);
    let n_ncas = 10;

    let task = &dataset.get_task("36a08778").unwrap();
    let grids = task.test_inputs();
    let config = Config::default();

    let ncas = (0..n_ncas)
        .map(|_| {
            let mut nca = NCA::new(config.clone());
            nca.initialize_random(&mut rng);
            nca
        })
        .collect_vec();

    for _ in 0..1000 {
        let mut executor = PopNCAExecutorGpuBatch::new(ncas.clone(), &grids);
        executor.run();
    }
}
