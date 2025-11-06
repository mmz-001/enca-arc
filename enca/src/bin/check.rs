use clap::Parser;
use enca::dataset::Submission;
use enca::serde_utils::JSONReadWrite;
use indexmap::IndexMap;

type ARCGrid = Vec<Vec<u8>>;
type GTSolutions = IndexMap<String, Vec<ARCGrid>>;

#[derive(Parser, Debug)]
struct Args {
    /// Predicted solutions JSON file
    #[arg(short = 'p', long)]
    pred_path: String,
    /// Ground truth solutions JSON file
    #[arg(short = 'a', long)]
    gt_path: String,
}

fn main() {
    let args = Args::parse();
    let pred_path = args.pred_path;
    let gt_path = args.gt_path;

    let pred_submission: Submission = Submission::read_json(&pred_path)
        .unwrap_or_else(|e| panic!("Failed to read predicted solutions file '{}': {}", &pred_path, e));

    let gt_solutions: GTSolutions = <GTSolutions as JSONReadWrite>::read_json(&gt_path)
        .unwrap_or_else(|e| panic!("Failed to read ground truth solutions file '{}': {}", &gt_path, e));

    // Validate format: every GT task id must be present in submission,
    // and the number of predicted outputs per task must match GT count.
    for (task_id, gt_outputs) in &gt_solutions {
        let preds = pred_submission
            .get(task_id)
            .unwrap_or_else(|| panic!("Submission missing predictions for task_id '{}'", task_id));

        if preds.len() != gt_outputs.len() {
            panic!(
                "Submission for task_id '{}' has {} predictions but ground truth has {} outputs",
                task_id,
                preds.len(),
                gt_outputs.len()
            );
        }
    }

    // Compute score
    let mut correct: usize = 0;
    let mut total: usize = 0;

    for (task_id, gt_outputs) in &gt_solutions {
        let preds = &pred_submission[task_id];
        for (i, gt_grid) in gt_outputs.iter().enumerate() {
            let p = &preds[i];
            let hit = p.attempt_1 == *gt_grid || p.attempt_2 == *gt_grid;
            if hit {
                correct += 1;
            }
            total += 1;
        }
    }

    if total == 0 {
        println!("0 / 0 correct, accuracy = 0.0%");
        return;
    }

    let accuracy = correct as f64 / total as f64;
    println!("{} / {} correct, accuracy = {:.3}%", correct, total, accuracy * 100.0);
}
