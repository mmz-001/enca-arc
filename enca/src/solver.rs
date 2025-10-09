use core::f32;

use crate::config::Config;
use crate::env::{compute_fitness, eval};
use crate::executors::NCAEnsembleExecutor;
use crate::metrics::TrainOutput;
use crate::nca::NCAEnsemble;
use crate::selector::{Optimize, Score, TournamentSelector};
use crate::utils::mean;
use crate::{dataset::Task, nca::NCA};
use cmaes::{CMAESOptions, DVector, ObjectiveFunction};
use itertools::Itertools;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

pub fn train(task: &Task, verbose: bool, config: &Config, seed: u64) -> Vec<TrainOutput> {
    let selector = TournamentSelector::new(config.k, Optimize::Minimize);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let mut nca = NCA::new(config.max_steps);
    nca.clear(); // Start with params zeroed

    let ncas = vec![nca];

    let initial_ensemble = NCAEnsemble::new(ncas, task.id.clone());

    let executors = task
        .train
        .iter()
        .map(|example| NCAEnsembleExecutor::new(initial_ensemble.clone(), &example.input))
        .collect_vec();

    let individual = IndividualState {
        executors,
        task: task.clone(),
        fitness: f32::INFINITY,
        config: config.clone(),
    };

    let mut population = vec![individual; config.pop];

    for n in 0..config.max_ncas {
        // Pre-generate independent seeds to avoid sharing RNG across threads
        let seeds: Vec<u64> = (0..population.len()).map(|_| rng.random()).collect();
        let cmaes_initial_sigma = config.cmaes_initial_sigma;
        let cmaes_max_fun_evals = config.cmaes_max_fun_evals;

        if verbose {
            println!("nca_id={n}")
        }

        population.par_iter_mut().enumerate().for_each(|(i, individual)| {
            if n + 1 > individual.executors[0].ensemble.ncas.len() {
                // Initialize with identity NCA
                let nca = NCA::new(config.max_steps);
                for executor in &mut individual.executors {
                    executor.upsert_nca(nca.clone(), n);
                }
            }

            let ensemble = &individual.executors.first().unwrap().ensemble;
            let initial_mean = ensemble.ncas.last().unwrap().to_vec();

            let mut cmaes_state = CMAESOptions::new(initial_mean, cmaes_initial_sigma.into())
                // .enable_printing(if verbose { Some(5000) } else { None })
                .tol_fun_hist(1e-7)
                .fun_target(1e-7)
                .seed(seeds[i])
                .max_function_evals(cmaes_max_fun_evals)
                .build(individual.clone())
                .unwrap();

            let results = cmaes_state.run();

            let overall_best = results.overall_best.unwrap();

            let cur_nca = NCA::from_vec(
                overall_best.point.iter().map(|v| *v as f32).collect_vec(),
                config.max_steps,
            );
            for executor in &mut individual.executors {
                executor.upsert_nca(cur_nca.clone(), n);
                executor.run();
            }
            individual.fitness = overall_best.value as f32;
        });

        if verbose {
            let fitnesses: Vec<f32> = population.iter().map(|ind| ind.fitness).collect();
            let best = fitnesses.iter().cloned().fold(f32::INFINITY, |a, b| a.min(b));
            let worst = fitnesses.iter().cloned().fold(f32::NEG_INFINITY, |a, b| a.max(b));
            let avg = mean(&fitnesses);
            println!(
                "Population summary: best={:.2e}, avg={:.2e}, worst={:.2e}",
                best, avg, worst
            );
        }

        if n == config.max_ncas - 1 {
            break;
        }

        population = selector.select(&population, &mut rng);
    }

    let mut train_ensembles = Vec::with_capacity(population.len());

    for individual in population {
        let ensemble = individual.executors[0].ensemble.clone();

        let accs = task
            .train
            .iter()
            .map(|example| eval(&example.input, &example.output, &ensemble))
            .collect_vec();

        train_ensembles.push(TrainOutput {
            ensemble,
            train_accs: accs,
            fitness: individual.fitness,
        });
    }

    train_ensembles.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());

    if verbose {
        let solved = train_ensembles
            .iter()
            .map(|ensemble| mean(&ensemble.train_accs) == 1.0)
            .filter(|s| *s)
            .count();

        println!(
            "Pop solved: count={}/{} pct: {:.3}%",
            solved,
            train_ensembles.len(),
            solved as f32 / train_ensembles.len() as f32 * 100.0
        )
    }

    train_ensembles
}

#[derive(Clone)]
struct IndividualState {
    task: Task,
    /// Executors for each example in the task.
    /// Iteration starts from the last executor to avoid recomputing previous executors.
    executors: Vec<NCAEnsembleExecutor>,
    fitness: f32,
    config: Config,
}

impl Score for IndividualState {
    fn score(&self) -> f32 {
        self.fitness
    }
}

impl ObjectiveFunction for IndividualState {
    fn evaluate(&mut self, x: &DVector<f64>) -> f64 {
        let nca = NCA::from_vec(x.iter().map(|v| *v as f32).collect_vec(), self.config.max_steps);

        let mut new_executors = self.executors.clone();
        for executor in &mut new_executors {
            executor.upsert_nca(nca.clone(), executor.ensemble.ncas.len() - 1);
        }

        mean(
            &self
                .task
                .train
                .iter()
                .zip(new_executors)
                .map(|(example, executor)| compute_fitness(example, &executor, &self.config))
                .collect_vec(),
        )
        .into()
    }
}

impl ObjectiveFunction for &mut IndividualState {
    fn evaluate(&mut self, x: &DVector<f64>) -> f64 {
        IndividualState::evaluate(*self, x)
    }
}
