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
    let dataset = Dataset::load(&tasks_path, None);

    let grid_small = &dataset.get_task("794b24be").unwrap().train[0].input; // 3x3 grid
    let grid_large = &dataset.get_task("264363fd").unwrap().train[0].input; // 30x30 grid

    let mut nca = NCA::new(100);
    nca.initialize_random(&mut rng);

    let mut group = c.benchmark_group("nca_single_grid");

    group.bench_function("small", |b| {
        b.iter(|| {
            let mut executor = NCAExecutor::new(nca.clone(), &grid_small, Backend::GPU);
            executor.run();
        })
    });

    group.bench_function("large", |b| {
        b.iter(|| {
            let mut executor = NCAExecutor::new(nca.clone(), &grid_large, Backend::GPU);
            executor.run();
        })
    });

    group.finish();
}

fn multi_grid(c: &mut Criterion) {
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let tasks_path = "./data/v2/arc-agi_evaluation_challenges.json";
    let dataset = Dataset::load(&tasks_path, None);
    let n_ncas = 10;

    let task = &dataset.get_task("36a08778").unwrap();
    let grids = task.test_inputs();

    let ncas = (0..n_ncas)
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

    // group.bench_function("cpu", |b| {
    //     b.iter(|| {
    //         for nca in &ncas {
    //             for grid in &grids {
    //                 let mut executor = NCAExecutor::new(nca.clone(), grid, Backend::CPU);
    //                 executor.run();
    //             }
    //         }
    //     })
    // });

    group.finish();
}

criterion_group!(benches, single_grid, multi_grid);
criterion_main!(benches);
