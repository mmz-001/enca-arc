use std::time::Instant;

use clap::Parser;
use enca::augment::augment;
use enca::config::Config;
use enca::criteria::train_preserves_grid_size;
use enca::dataset::{Submission, TestSubmissionOutput};
use enca::env::inference;
use enca::executors::Backend;
use enca::executors::gpu::CUDA;
use enca::serde_utils::JSONReadWrite;
use enca::utils::mean;
use enca::voting::vote;
use enca::{dataset::Dataset, solver::train};
use indexmap::IndexMap;
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;

#[derive(Parser, Debug)]
struct Args {
    /// Tasks JSON file
    #[arg(short = 't', long)]
    tasks_path: String,
    /// Seed for reproducibility
    #[arg(short = 's', long)]
    seed: u64,
    /// Config file path
    #[arg(short = 'c', long)]
    config_path: Option<String>,
}

fn main() {
    let args = Args::parse();
    let tasks_path = args.tasks_path;
    let seed = args.seed;
    let dataset = Dataset::load(&tasks_path, None);
    let submission_path = "./submission.json";
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

    println!("Loaded tasks from '{}': tasks={}", tasks_path, dataset.tasks.len());

    let start = Instant::now();

    let tasks = dataset.tasks;

    let pb = ProgressBar::new(tasks.len() as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}<{eta_precise}] {bar:40.cyan/blue} {pos}/{len} {percent:>3}% {per_sec} it/s",
        )
        .unwrap()
        .progress_chars("##-"),
    );

    let results: Vec<(String, Vec<TestSubmissionOutput>)> = tasks
        .iter()
        .map(|task| {
            pb.inc(1);
            let default_output = (task.id.clone(), vec![TestSubmissionOutput::default(); task.test.len()]);
            if !train_preserves_grid_size(task) {
                return default_output;
            }

            let train_result = train(task, false, &config, seed);

            let mut test_submission_outputs: Vec<TestSubmissionOutput> = Vec::with_capacity(task.test.len());

            let solved_train = train_result
                .clone()
                .into_iter()
                .filter(|result| mean(&result.train_accs) == 1.0)
                .collect_vec();

            if solved_train.is_empty() {
                return default_output;
            }

            let selected_train = if solved_train.is_empty() {
                train_result
            } else {
                solved_train
            };

            for input in task.test_inputs() {
                let aug_enca = selected_train
                    .iter()
                    .map(|result| augment(input, task, result.nca.clone(), seed, &config))
                    .collect_vec();
                let top_k_aug_ncas = vote(input, &aug_enca, 2, false, config.backend.clone());

                let pred_grid = inference(input, &top_k_aug_ncas[0], config.backend.clone());

                let attempt_1 = pred_grid.data().clone();
                let attempt_2 = if top_k_aug_ncas.len() >= 2 {
                    let pred_grid = inference(input, &top_k_aug_ncas[1], config.backend.clone());
                    pred_grid.data().clone()
                } else {
                    attempt_1.clone()
                };

                test_submission_outputs.push(TestSubmissionOutput { attempt_1, attempt_2 });
            }
            (task.id.clone(), test_submission_outputs)
        })
        .collect();

    pb.finish_and_clear();
    let total_elapsed_ms = start.elapsed().as_millis();
    println!("elapsed_ms={}", total_elapsed_ms);

    let submission: Submission = IndexMap::from_iter(results);

    submission
        .write_json(submission_path)
        .unwrap_or_else(|e| panic!("Failed to write submission file '{}': {}", &submission_path, e));

    println!("Wrote submission to -> {}", submission_path);
}
