use std::collections::HashMap;
use std::time::Instant;

use clap::Parser;
use enca::augment::augment;
use enca::config::Config;
use enca::criteria::train_preserves_grid_size;
use enca::dataset::{Submission, TestSubmissionOutput};
use enca::env::inference;
use enca::serde_utils::JSONReadWrite;
use enca::utils::mean;
use enca::voting::vote;
use enca::{dataset::Dataset, solver::train};
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use rayon::prelude::*;

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

    println!("Loaded tasks from '{}': tasks={}", tasks_path, dataset.tasks.len());
    let start = Instant::now();

    let tasks = dataset.tasks;

    let config = if let Some(config_path) = args.config_path {
        Config::read_json(&config_path)
            .unwrap_or_else(|e| panic!("Failed to read config file '{}': {}", &config_path, e))
    } else {
        Config::default()
    };

    let pb = ProgressBar::new(tasks.len() as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}<{eta_precise}] {bar:40.cyan/blue} {pos}/{len} {percent:>3}% {per_sec} it/s",
        )
        .unwrap()
        .progress_chars("##-"),
    );

    let results: Vec<(String, Vec<TestSubmissionOutput>)> = tasks
        .par_iter()
        .map(|task| {
            let output = if train_preserves_grid_size(task) {
                let train_result = train(task, false, &config, seed);

                let mut test_submission_outputs: Vec<TestSubmissionOutput> = Vec::with_capacity(task.test.len());

                let solved_train = train_result
                    .clone()
                    .into_iter()
                    .filter(|result| mean(&result.train_accs) == 1.0)
                    .collect_vec();

                let selected_train_result = if !solved_train.is_empty() {
                    solved_train
                } else {
                    // Fallback
                    train_result
                };

                for input in task.test_inputs() {
                    let aug_ensembles = selected_train_result
                        .iter()
                        .map(|result| augment(input, task, result.ensemble.clone(), seed))
                        .collect_vec();
                    let top_k_aug_ensembles = vote(input, &aug_ensembles, 2, false);

                    let pred_grid = inference(input, &top_k_aug_ensembles[0]);

                    let attempt_1 = pred_grid.data().clone();
                    let attempt_2 = if top_k_aug_ensembles.len() >= 2 {
                        let pred_grid = inference(input, &top_k_aug_ensembles[1]);
                        pred_grid.data().clone()
                    } else {
                        attempt_1.clone()
                    };

                    test_submission_outputs.push(TestSubmissionOutput { attempt_1, attempt_2 });
                }

                test_submission_outputs
            } else {
                vec![TestSubmissionOutput::default(); task.test.len()]
            };
            pb.inc(1);
            (task.id.clone(), output)
        })
        .collect();

    pb.finish_and_clear();
    let total_elapsed_ms = start.elapsed().as_millis();
    println!("elapsed_ms={}", total_elapsed_ms);

    let submission: Submission = HashMap::from_iter(results);

    submission
        .write_json(submission_path)
        .unwrap_or_else(|e| panic!("Failed to write submission file '{}': {}", &submission_path, e));

    println!("Wrote submission to -> {}", submission_path);
}
