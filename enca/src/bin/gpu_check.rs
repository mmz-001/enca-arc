use enca::{
    dataset::Dataset,
    executors::{Backend, NCAExecutor, gpu::PopNCAExecutorGpuBatch},
    nca::NCA,
};
use itertools::Itertools;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;

fn main() {
    let tasks_path = "./data/v1/arc-agi_training_challenges.json";
    let train_dataset = Dataset::load(tasks_path, None);
    let mut rng = ChaCha12Rng::seed_from_u64(1);
    let pop_size = 4;

    for task in &train_dataset.tasks {
        let max_steps = rng.random_range(1..=120);
        let pop_ncas = (0..pop_size).map(|_| random_nca(&mut rng, max_steps)).collect_vec();

        let mut pop_gpu_executor =
            PopNCAExecutorGpuBatch::new(pop_ncas.clone(), &task.train_inputs().into_iter().collect_vec());

        pop_gpu_executor.run();

        for (ind_idx, nca) in pop_ncas.iter().enumerate() {
            for (idx, input) in task.train_inputs().iter().enumerate() {
                let backend = Backend::CPU;
                let mut cpu_executor = NCAExecutor::new(nca.clone(), input, backend.clone());
                cpu_executor.run();
                let cpu_hash = cpu_executor.substrate().to_grid().get_hash();
                let gpu_hash = pop_gpu_executor.individuals[ind_idx].substrates[idx]
                    .to_grid()
                    .get_hash();

                if cpu_hash != gpu_hash {
                    panic!(
                        "Hash mismatch on task {} idx, ind idx {}, input grid id {}. grid width = {}, height = {}.\nCPU: hash={}\nGPU: hash={}",
                        task.id,
                        ind_idx,
                        idx,
                        input.width(),
                        input.height(),
                        cpu_hash,
                        gpu_hash,
                    );
                }
            }
        }
    }

    println!("GPU and CPU results match exactly!")
}

fn random_nca(rng: &mut impl Rng, max_steps: usize) -> NCA {
    let mut nca = NCA::new(max_steps);
    nca.initialize_random(rng);
    nca
}
