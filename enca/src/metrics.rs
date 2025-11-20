use serde::{Deserialize, Serialize};

use crate::nca::NCA;

#[derive(Debug, Serialize, Deserialize)]
pub struct TaskReport {
    pub task_id: String,
    pub n_examples_train: usize,
    pub n_examples_test: usize,
    pub train_accs: Vec<f32>,
    pub test_accs: Vec<f32>,
    pub duration_ms: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OverallSummary {
    pub n_tasks: usize,
    pub total_test_grids: usize,
    pub total_test_correct: usize,
    pub test_accuracy: f32,
    pub elapsed_ms: u128,
    pub seed: u64,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct TrainIndividual {
    pub nca: NCA,
    pub train_accs: Vec<f32>,
    pub fitness: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndividualMetrics {
    pub id: usize,
    pub mean_acc: f32,
    pub fitness: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub individual_metrics: Vec<IndividualMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainMetrics {
    pub epoch_metrics: Vec<EpochMetrics>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct TrainOutput {
    pub population: Vec<TrainIndividual>,
    pub metrics: TrainMetrics,
}
