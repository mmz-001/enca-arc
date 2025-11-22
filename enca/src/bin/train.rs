use std::fs;
use std::time::Instant;

use clap::Parser;
use enca::augment::{TaskNCAs, augment};
use enca::config::Config;
use enca::criteria::train_preserves_grid_size;
use enca::executors::Backend;
use enca::executors::gpu::CUDA;
use enca::metrics::{OverallSummary, TaskReport};
use enca::serde_utils::JSONReadWrite;
use enca::utils::{mean, timestamp_for_dir};
use enca::voting::vote;
use enca::{dataset::Dataset, env::eval, solver::train};
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

struct TestOutcome {
    count: usize,
    correct: usize,
}

#[derive(Parser, Debug)]
struct Args {
    /// Task ID for running on a single task
    #[arg(short = 'i', long)]
    id: Option<String>,
    /// Tasks JSON file
    #[arg(short = 't', long)]
    tasks_path: String,
    /// Solutions JSON file for evaluation
    #[arg(short = 'a', long)]
    solutions_path: String,
    /// Run output directory. Defaults to a timestamped directory in runs/
    #[arg(short = 'r', long)]
    out_dir: Option<String>,
    /// Seed for reproducibility
    #[arg(short = 's', long)]
    seed: Option<u64>,
    /// Config file path
    #[arg(short = 'c', long)]
    config_path: Option<String>,
}

fn main() {
    let args = Args::parse();
    let tasks_path = args.tasks_path;
    let solutions_path = args.solutions_path;
    let verbose = args.id.is_some();
    let config = if let Some(config_path) = args.config_path {
        Config::read_json(&config_path)
            .unwrap_or_else(|e| panic!("Failed to read config file '{}': {}", &config_path, e))
    } else {
        Config::default()
    };

    // Initialize GPUs
    if config.backend == Backend::GPU {
        _ = &*CUDA;
    }

    let seed = if let Some(seed) = args.seed {
        seed
    } else {
        rand::random()
    };

    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let dataset = Dataset::load(&tasks_path, Some(&solutions_path));
    println!(
        "Loaded tasks from '{}' and solutions from '{}': tasks={}",
        tasks_path,
        solutions_path,
        dataset.tasks.len()
    );

    let out_dir = if let Some(out_dir) = args.out_dir {
        out_dir
    } else {
        let timestamp = timestamp_for_dir();
        format!("runs/{timestamp}")
    };

    let metrics_dir = format!("{out_dir}/metrics");
    let model_dir = format!("{out_dir}/models");

    fs::create_dir_all(&out_dir).unwrap_or_else(|e| panic!("Failed to create out_dir '{}': {}", out_dir, e));
    fs::create_dir_all(&metrics_dir)
        .unwrap_or_else(|e| panic!("Failed to create metrics_dir '{}': {}", metrics_dir, e));
    fs::create_dir_all(&model_dir).unwrap_or_else(|e| panic!("Failed to create model_dir: {}", e));

    if let Some(id) = &args.id {
        println!("Running train for task with id : {}", id);
    }

    let tasks = if let Some(id) = &args.id {
        vec![
            dataset
                .get_task(id)
                .unwrap_or_else(|| panic!("Task with id={id} not found"))
                .clone(),
        ]
    } else {
        dataset.tasks.clone()
    };

    let solutions = if let Some(id) = &args.id {
        vec![
            dataset
                .get_solution(id)
                .unwrap_or_else(|| panic!("Solution with id={id} not found"))
                .clone(),
        ]
    } else {
        dataset.solutions.unwrap()
    };

    let tasks_and_solutions = tasks.iter().zip(&solutions).collect_vec();

    let start = Instant::now();

    let total = tasks.len() as u64;

    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}<{eta_precise}] {bar:40.cyan/blue} {pos}/{len} {percent:>3}% {per_sec} it/s",
        )
        .unwrap()
        .progress_chars("##-"),
    );

    let results: Vec<TestOutcome> = tasks_and_solutions
        .iter()
        .map(|(task, solution)| {
            let start = Instant::now();
            if !verbose {
                pb.inc(1);
            }

            let task_id = &task.id;

            let default_outcome = TestOutcome {
                count: task.test.len(),
                correct: 0,
            };

            // Test task io shapes match when train task shapes are all the same.
            // We check this property for all data in `assertions.rs`
            if !train_preserves_grid_size(task) {
                return default_outcome;
            }

            let train_output = train(task, verbose, &config, &mut rng);

            if verbose {
                if let Err(e) = enca::plotting::plot_metrics(&train_output.metrics, &out_dir, task_id) {
                    eprintln!("Failed to plot metrics for task {}: {}", task_id, e);
                }
            }

            let train_result = train_output.population;

            let best_train_result = train_result[0].clone();
            let train_accs = best_train_result.train_accs;

            let mut test_ncas = Vec::with_capacity(task.test.len());

            let mut test_accs = Vec::with_capacity(solution.outputs.len());
            let solved_train = train_result
                .clone()
                .into_iter()
                .filter(|result| mean(&result.train_accs) == 1.0)
                .collect_vec();

            let selected_train = if solved_train.is_empty() {
                train_result
            } else {
                solved_train
            };

            for (input, output) in task.test_inputs().iter().zip(&solution.outputs) {
                let aug_ncas = selected_train
                    .iter()
                    .map(|result| augment(input, task, result.nca.clone(), &config, &mut rng))
                    .collect_vec();
                let top_k_aug_ncas = vote(input, &aug_ncas, 2, verbose, config.backend.clone());

                let top_aug_nca = if top_k_aug_ncas.len() >= 2 {
                    let attempt_1_acc = eval(input, output, &top_k_aug_ncas[0], config.backend.clone());
                    let attempt_2_acc = eval(input, output, &top_k_aug_ncas[1], config.backend.clone());
                    if attempt_1_acc > attempt_2_acc {
                        &top_k_aug_ncas[0]
                    } else {
                        &top_k_aug_ncas[1]
                    }
                } else {
                    &top_k_aug_ncas[0]
                };

                test_accs.push(eval(input, output, top_aug_nca, config.backend.clone()));
                test_ncas.push(top_aug_nca.clone());
            }

            let elapsed = start.elapsed().as_millis();

            let task_ncas = TaskNCAs {
                train: best_train_result.nca,
                test: test_ncas,
            };

            let train_mean = mean(&train_accs);
            let test_mean = mean(&test_accs);

            if verbose {
                println!("\n==> Task {}", task_id);
                println!("train_accs(%)={:?} | mean={:.5}", &train_accs, train_mean);
                println!("test_accs(%)={:?} | mean={:.5}", test_accs, test_mean);
            }

            let nca_path = format!("{model_dir}/{task_id}.json");
            task_ncas.write_json(&nca_path).unwrap();

            let report = TaskReport {
                task_id: task_id.clone(),
                n_examples_train: task.train.len(),
                n_examples_test: task.test.len(),
                train_accs: train_accs.clone(),
                test_accs: test_accs.clone(),
                duration_ms: Some(elapsed as usize),
            };
            let metrics_path = format!("{metrics_dir}/{task_id}.json");
            report
                .write_json(&metrics_path)
                .unwrap_or_else(|e| panic!("Failed to create metrics file '{}': {}", metrics_path, e));

            let test_correct = test_accs.iter().filter(|acc| **acc == 1.0).collect_vec().len();

            TestOutcome {
                count: task.test.len(),
                correct: test_correct,
            }
        })
        .collect();

    if !verbose {
        pb.finish();
    }

    let count: usize = results.iter().map(|r| r.count).sum();
    let test_correct: usize = results.iter().map(|r| r.correct).sum();
    let n_tasks: usize = results.len();

    let total_elapsed_ms = start.elapsed().as_millis();
    let test_accuracy = test_correct as f32 / count as f32 * 100.0;

    let summary = OverallSummary {
        n_tasks,
        total_test_grids: count,
        total_test_correct: test_correct,
        test_accuracy,
        elapsed_ms: total_elapsed_ms,
        seed,
    };

    let summary_path = format!("{out_dir}/summary.json");
    summary
        .write_json(&summary_path)
        .unwrap_or_else(|e| panic!("Failed to create summary file '{}': {}", summary_path, e));

    let config_path = format!("{out_dir}/config.json");
    config
        .write_json(&config_path)
        .unwrap_or_else(|e| panic!("Failed to create config file '{}': {}", config_path, e));

    println!("==== Overall Summary ====");
    println!("tasks={}, total_test_grids={}", n_tasks, count);
    println!("total_test_correct={}", test_correct);
    println!("test_accuracy={:.2}%", test_accuracy);
    println!("elapsed_ms={}", total_elapsed_ms);
    println!("Metrics summary -> {}", summary_path);
}
