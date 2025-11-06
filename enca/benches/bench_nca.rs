use criterion::{Criterion, criterion_group, criterion_main};
use enca::{
    augment::TaskNCAs,
    dataset::Dataset,
    executors::{Backend, NCAExecutor},
    serde_utils::JSONReadWrite,
};

fn nca_run(c: &mut Criterion) {
    let nca_dir = "./benches/models";
    let tasks_path = "./data/v1/arc-agi_training_challenges.json";
    let train_dataset = Dataset::load(&tasks_path, None);

    let task_ncas = TaskNCAs::load(&nca_dir).unwrap_or_else(|e| {
        panic!("Failed to read dataset root '{}': {}", nca_dir, e);
    });

    let task = &train_dataset.get_task("264363fd").unwrap(); // 30x30 grid

    let nca = &task_ncas.iter().find(|(id, _)| *id == task.id).unwrap().1.train;

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
