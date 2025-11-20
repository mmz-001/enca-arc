use crate::config::Config;
use crate::constants::{BIASES_RNG, N_PARAMS, WEIGHTS_RNG};
use crate::env::{compute_fitness_pop, eval};
use crate::lmcma::LMCMAOptions;
use crate::metrics::{EpochMetrics, IndividualMetrics, TrainIndividual, TrainMetrics, TrainOutput};
use crate::selector::{Optimize, Score, TournamentSelector};
use crate::utils::mean;
use crate::{dataset::Task, nca::NCA};
use cmaes::DVector;
use cmaes::objective_function::BatchObjectiveFunction;
use core::f32;
use itertools::Itertools;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

pub fn train(task: &Task, verbose: bool, config: &Config, seed: u64) -> TrainOutput {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let selector = TournamentSelector::new(config.k, Optimize::Maximize);
    let mut indexer = 0;

    let mut population = (0..config.pop)
        .map(|_| IndividualState::new(&mut indexer, task.clone(), config.clone()))
        .collect_vec();

    let mut solved: Vec<IndividualState> = vec![];

    // Pre-generate seeds
    let mut metrics = TrainMetrics { epoch_metrics: vec![] };

    for epoch in 0..config.epochs {
        if verbose {
            print!("epoch={epoch:03}, ");
        }

        if epoch.is_multiple_of(20) {
            population = selector
                .select(&population.iter().collect_vec(), &mut rng)
                .into_iter()
                .cloned()
                .collect_vec();
        }

        if population.len() < config.pop {
            let deficit = config.pop - population.len();

            for _ in 0..deficit {
                population.push(IndividualState::new(&mut indexer, task.clone(), config.clone()));
            }
        }

        let seed = rng.random();
        solve_pop(&mut population, task, seed, config.clone(), None);

        let (unsolved, epoch_solved): (Vec<_>, Vec<_>) =
            population.iter().partition(|individual| individual.mean_acc < 1.0);

        solved.extend(epoch_solved.into_iter().cloned());

        if verbose {
            let best_fitness = population
                .iter()
                .map(|ind| ind.fitness)
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();

            let best_acc = population
                .iter()
                .map(|ind| ind.mean_acc)
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();

            println!(
                "solved count={}, pop best: fitness={:.3e}, accuracy={:.3}",
                solved.len(),
                best_fitness,
                best_acc,
            );

            metrics.epoch_metrics.push(EpochMetrics {
                epoch,
                individual_metrics: population
                    .iter()
                    .map(|ind| IndividualMetrics {
                        id: ind.id,
                        fitness: ind.fitness,
                        mean_acc: ind.mean_acc,
                    })
                    .collect_vec(),
            });
        }

        population = unsolved.into_iter().cloned().collect_vec();
    }

    println!("Optimizing solved...");

    let mut final_solve_config = config.clone();
    final_solve_config.subset_size = N_PARAMS;
    final_solve_config.max_fun_evals = config.max_fun_evals * 50;

    let seed = rng.random();
    solve_pop(&mut solved, task, seed, config.clone(), Some(N_PARAMS));

    population.extend(solved);
    let mut train_ncas = Vec::with_capacity(population.len());

    for individual in population {
        let accs = task
            .train
            .iter()
            .map(|example| eval(&example.input, &example.output, &individual.nca, config.backend.clone()))
            .collect_vec();

        let fitness = individual.fitness;

        train_ncas.push(TrainIndividual {
            nca: individual.nca,
            train_accs: accs,
            fitness,
        });
    }

    train_ncas.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
    train_ncas.sort_by(|b, a| mean(&a.train_accs).partial_cmp(&mean(&b.train_accs)).unwrap());

    TrainOutput {
        population: train_ncas,
        metrics,
    }
}

fn solve_pop(population: &mut Vec<IndividualState>, task: &Task, seed: u64, config: Config, lambda: Option<usize>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let seeds: Vec<u64> = (0..population.len()).map(|_| rng.random()).collect();

    population.par_iter_mut().enumerate().for_each(|(i, individual)| {
        let mut new_individual = individual.clone();
        let new_nca = &mut new_individual.nca;

        let mut rng = ChaCha8Rng::seed_from_u64(seeds[i]);

        let mut idxs = (0..N_PARAMS).collect_vec();
        idxs.shuffle(&mut rng);

        new_individual.train_param_idxs = idxs[0..(config.subset_size).min(idxs.len())].to_vec();

        let all_params = new_nca.to_vec();
        let initial_mean: Vec<f64> = new_individual
            .train_param_idxs
            .iter()
            .map(|&i| all_params[i] as f64)
            .collect();

        let mut es_state_builder = LMCMAOptions::new(initial_mean, config.initial_sigma)
            .tol_fun_hist(1e-12)
            .fun_target(1e-7)
            .seed(seeds[i])
            .max_function_evals(config.max_fun_evals);
        es_state_builder.lambda = lambda;

        let mut es_state = es_state_builder.build(new_individual.clone()).unwrap();

        let results = es_state.run_batch();

        let overall_best = results.overall_best.unwrap();

        let point = overall_best.point.clone();
        let fitness = overall_best.value;

        let new_nca = construct_nca(&new_individual, &point);

        let accs = task
            .train
            .iter()
            .map(|example| eval(&example.input, &example.output, &new_nca, config.backend.clone()))
            .collect_vec();

        let mean_acc = mean(&accs);

        if mean_acc >= individual.mean_acc {
            individual.nca = new_nca;
            individual.fitness = fitness as f32;
            individual.mean_acc = mean_acc;
            individual.train_param_idxs = new_individual.train_param_idxs;
        }
    });
}

#[derive(Clone)]
struct IndividualState {
    id: usize,
    task: Task,
    nca: NCA,
    fitness: f32,
    mean_acc: f32,
    config: Config,
    train_param_idxs: Vec<usize>,
}

impl IndividualState {
    fn new(indexer: &mut usize, task: Task, config: Config) -> Self {
        let id = indexer.clone();
        *indexer += 1;

        let nca = NCA::new(config.max_steps);

        IndividualState {
            id,
            nca,
            task,
            fitness: f32::INFINITY,
            config,
            train_param_idxs: vec![],
            mean_acc: 0.0,
        }
    }
}

fn construct_nca(individual: &IndividualState, x: &DVector<f64>) -> NCA {
    let mut all_params = individual.nca.to_vec();

    for (j, idx) in individual.train_param_idxs.iter().enumerate() {
        all_params[*idx] = x[j] as f32;
    }

    NCA::from_vec(
        &all_params[WEIGHTS_RNG],
        &all_params[BIASES_RNG],
        individual.config.max_steps,
    )
}

impl BatchObjectiveFunction for IndividualState {
    fn evaluate_batch(&self, xs: &[DVector<f64>]) -> Vec<f64> {
        let ncas = xs.iter().map(|x| construct_nca(self, x)).collect_vec();
        compute_fitness_pop(&self.task.train, ncas, &self.config)
    }
}

impl BatchObjectiveFunction for &mut IndividualState {
    fn evaluate_batch(&self, x: &[DVector<f64>]) -> Vec<f64> {
        IndividualState::evaluate_batch(self, x)
    }
}

impl Score for IndividualState {
    fn score(&self) -> f32 {
        self.mean_acc
    }
}
