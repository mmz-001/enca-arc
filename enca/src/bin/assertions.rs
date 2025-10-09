/*!
 * Inference time augmentations and grid sizes used by NCAs assume certain properties and invariants hold for
 * all ARC 1 and 2 problems. For each assertion, we assume a certain property holds for the task by looking only at the
 * train examples and test input examples and assert the property holds for the train output examples.
 */

use clap::Parser;
use enca::{criteria::train_preserves_grid_size, dataset::Dataset};

/// If:
/// - For every train example, the input size equals the output size.
///
/// Then:
/// - For every test example, the input size equals the output size.
fn assert_test_preserves_grid_size_when_train_does(dataset: &Dataset, verbose: bool) {
    let total = dataset.tasks.len();
    let mut task_premise_count = 0;
    let mut grids_premise_count = 0;
    let test_grids = dataset.tasks.iter().map(|task| task.test_inputs().len()).sum::<usize>();

    for (task, solution) in dataset.tasks.iter().zip(dataset.solutions.as_ref().unwrap()) {
        if train_preserves_grid_size(task) {
            task_premise_count += 1;
            grids_premise_count += task.test.len();

            for (input, output) in task.test.iter().map(|x| &x.input).zip(&solution.outputs) {
                assert!(
                    input.shape() == output.shape(),
                    "When every train example preserves grid size, each test example must also preserve size. Task: {} example: {}. Input size: {:?}, Output size: {:?}",
                    task.id,
                    task.id,
                    input.shape(),
                    output.shape()
                );
            }
        }
    }

    if verbose {
        let pct = (task_premise_count as f64 * 100.0) / (total as f64);
        println!(
            "Train input/output sizes equal for {}/{} tasks ({:.1}%)",
            task_premise_count, total, pct
        );
        let pct = (grids_premise_count as f64 * 100.0) / (test_grids as f64);
        println!(
            "Train input/output sizes equal for {}/{} test grids ({:.1}%)",
            grids_premise_count, test_grids, pct
        );
    }
}

#[derive(Parser, Debug)]
struct Args {
    #[arg()]
    v1_dir: Option<String>,
    #[arg()]
    v2_dir: Option<String>,
    #[arg(short, long, default_value_t = false, help = "Print verbose assertion summaries")]
    verbose: bool,
}

fn main() {
    let args = Args::parse();
    let v1_dir = args.v1_dir.unwrap_or("./data/v1".to_owned());
    let v2_dir = args.v2_dir.unwrap_or("./data/v2".to_owned());

    let datasets = vec![
        (
            "ARC-AGI 1 Public Train",
            format!("{v1_dir}/arc-agi_training_challenges.json"),
            format!("{v1_dir}/arc-agi_training_solutions.json"),
        ),
        (
            "ARC-AGI 1 Public Evaluation",
            format!("{v1_dir}/arc-agi_evaluation_challenges.json"),
            format!("{v1_dir}/arc-agi_evaluation_solutions.json"),
        ),
        (
            "ARC-AGI 2 Public Train",
            format!("{v2_dir}/arc-agi_training_challenges.json"),
            format!("{v2_dir}/arc-agi_training_solutions.json"),
        ),
        (
            "ARC-AGI 2 Public Evaluation",
            format!("{v2_dir}/arc-agi_evaluation_challenges.json"),
            format!("{v2_dir}/arc-agi_evaluation_solutions.json"),
        ),
    ];

    let loaded: Vec<(String, Dataset)> = datasets
        .into_iter()
        .map(|(name, tasks_path, solutions_path)| {
            let dataset = Dataset::load(&tasks_path, Some(&solutions_path));
            (name.to_string(), dataset)
        })
        .collect();

    for (name, dataset) in &loaded {
        if args.verbose {
            println!("========{name}=========");
        }
        assert_test_preserves_grid_size_when_train_does(dataset, args.verbose);

        if args.verbose {
            println!("======================")
        }
    }
}
