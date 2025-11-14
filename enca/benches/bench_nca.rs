use criterion::{Criterion, criterion_group, criterion_main};
use enca::{
    dataset::Dataset,
    executors::{Backend, NCAExecutor},
    nca::NCA,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn nca_run(c: &mut Criterion) {
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let tasks_path = "./data/v1/arc-agi_training_challenges.json";
    let train_dataset = Dataset::load(&tasks_path, None);

    let task = &train_dataset.get_task("264363fd").unwrap(); // 30x30 grid

    let mut nca = NCA::new(100);
    nca.initialize_random(&mut rng);

    let mut group = c.benchmark_group("nca_run");

    let backend = Backend::GPU;

    let mut executor = NCAExecutor::new(nca.clone(), &task.train[0].input, backend.clone());
    executor.run();

    println!("output_grid_hash = {}", executor.substrate().to_grid().get_hash());
    group.bench_function("nca_run", |b| {
        b.iter(|| {
            let mut executor = NCAExecutor::new(nca.clone(), &task.train[0].input, backend.clone());
            executor.run();
        })
    });

    group.finish();
}

criterion_group!(benches, nca_run);
criterion_main!(benches);
