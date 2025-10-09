use criterion::{Criterion, criterion_group, criterion_main};
use enca::{augment::TaskECAEnsembles, dataset::Dataset, executors::NCAEnsembleExecutor, serde_utils::JSONReadWrite};

fn nca_run(c: &mut Criterion) {
    let nca_dir = "./benches/models";
    let tasks_path = "./data/v1/arc-agi_training_challenges.json";
    let train_dataset = Dataset::load(&tasks_path, None);

    let task_ensembles = TaskECAEnsembles::load(&nca_dir).unwrap_or_else(|e| {
        panic!("Failed to read dataset root '{}': {}", nca_dir, e);
    });

    let task = &train_dataset.get_task("264363fd").unwrap(); // 30x30 grid

    let ensemble = &task_ensembles
        .iter()
        .find(|multi_nca| multi_nca.0 == task.id)
        .unwrap()
        .1
        .train;

    let mut group = c.benchmark_group("nca_run");

    let mut executor = NCAEnsembleExecutor::new(ensemble.clone(), &task.train[0].input);
    executor.run();

    println!(
        "output_grid_hash = {}",
        executor.executors.last().unwrap().substrate.to_grid().get_hash()
    );
    group.bench_function("nca_run", |b| {
        b.iter(|| {
            let mut executor = NCAEnsembleExecutor::new(ensemble.clone(), &task.train[0].input);
            executor.run();
        })
    });

    group.finish();
}

criterion_group!(benches, nca_run);
criterion_main!(benches);
