use criterion::{Criterion, criterion_group, criterion_main};
use enca::{
    dataset::Dataset,
    executors::{Backend, NCAExecutor, gpu::PopNCAExecutorGpuBatch},
    nca::NCA,
};
use itertools::Itertools;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn single_grid(c: &mut Criterion) {
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let tasks_path = "./data/v1/arc-agi_training_challenges.json";
    let train_dataset = Dataset::load(&tasks_path, None);

    let task = &train_dataset.get_task("264363fd").unwrap(); // 30x30 grid

    let mut nca = NCA::new(100);
    nca.initialize_random(&mut rng);

    let mut group = c.benchmark_group("nca_single_grid");

    group.bench_function("gpu", |b| {
        b.iter(|| {
            let mut executor = NCAExecutor::new(nca.clone(), &task.train[0].input, Backend::GPU);
            executor.run();
        })
    });

    group.bench_function("cpu", |b| {
        b.iter(|| {
            let mut executor = NCAExecutor::new(nca.clone(), &task.train[0].input, Backend::CPU);
            executor.run();
        })
    });

    group.finish();
}

fn multi_grid(c: &mut Criterion) {
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let tasks_path = "./data/v2/arc-agi_evaluation_challenges.json";
    let train_dataset = Dataset::load(&tasks_path, None);
    let n_grids = 10;

    let tasks = &train_dataset.tasks[0..n_grids];
    let grids = tasks.iter().map(|task| &task.train[0].input).collect_vec();

    let ncas = (0..n_grids)
        .map(|_| {
            let mut nca = NCA::new(100);
            nca.initialize_random(&mut rng);
            nca
        })
        .collect_vec();

    let mut group = c.benchmark_group("nca_multi_grid");

    group.bench_function("gpu", |b| {
        b.iter(|| {
            let mut executor = PopNCAExecutorGpuBatch::new(ncas.clone(), &grids);
            executor.run();
        })
    });

    group.bench_function("cpu", |b| {
        b.iter(|| {
            for i in 0..n_grids {
                let nca = ncas[i].clone();
                let grid = &grids[i];
                let mut executor = NCAExecutor::new(nca, grid, Backend::CPU);
                executor.run();
            }
        })
    });

    group.finish();
}

criterion_group!(benches, single_grid, multi_grid);
criterion_main!(benches);
