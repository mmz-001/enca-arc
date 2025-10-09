use std::collections::HashMap;

use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::{grid::Grid, serde_utils::JSONReadWrite};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseTrainExample<T> {
    pub input: T,
    pub output: T,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseTestProblem<T> {
    pub input: T,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseTask<T> {
    pub id: String,
    pub train: Vec<BaseTrainExample<T>>,
    pub test: Vec<BaseTestProblem<T>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseSolution<T> {
    pub id: String,
    pub outputs: Vec<T>,
}

pub type TrainExample = BaseTrainExample<Grid>;
pub type TestProblem = BaseTestProblem<Grid>;
pub type Task = BaseTask<Grid>;
pub type Solution = BaseSolution<Grid>;

pub type ARCGrid = Vec<Vec<u8>>;
type ARCTestSolution = Vec<ARCGrid>;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ARCTask {
    pub train: Vec<BaseTrainExample<ARCGrid>>,
    pub test: Vec<BaseTestProblem<ARCGrid>>,
}

#[derive(Debug, Clone)]
pub struct Dataset {
    pub tasks: Vec<Task>,
    pub solutions: Option<Vec<Solution>>,
}

impl Dataset {
    pub fn load(tasks_path: &str, solutions_path: Option<&str>) -> Self {
        let tasks_by_id: HashMap<String, ARCTask> =
            <HashMap<String, ARCTask> as JSONReadWrite>::read_json(tasks_path).expect("Failed to read tasks JSON");

        let mut tasks: Vec<Task> = tasks_by_id
            .into_iter()
            .map(|(id, arc_task)| {
                let train = arc_task
                    .train
                    .into_iter()
                    .map(|ex| TrainExample {
                        input: Grid::from_vec(ex.input),
                        output: Grid::from_vec(ex.output),
                    })
                    .collect();

                let test = arc_task
                    .test
                    .into_iter()
                    .map(|p| TestProblem {
                        input: Grid::from_vec(p.input),
                    })
                    .collect();

                Task { id, train, test }
            })
            .collect();

        tasks.sort_by(|a, b| a.id.cmp(&b.id));

        // Load solutions if provided
        let solutions = solutions_path.map(|sol_path| {
            let sols_by_id: HashMap<String, ARCTestSolution> =
                <HashMap<String, ARCTestSolution> as JSONReadWrite>::read_json(sol_path)
                    .expect("Failed to read solutions JSON");

            let mut sols: Vec<Solution> = sols_by_id
                .into_iter()
                .map(|(id, arc_sol)| {
                    let test_outputs = arc_sol.into_iter().map(Grid::from_vec).collect();

                    Solution {
                        id,
                        outputs: test_outputs,
                    }
                })
                .collect();

            sols.sort_by(|a, b| a.id.cmp(&b.id));
            sols
        });

        Dataset { tasks, solutions }
    }

    pub fn get_task(&self, id: &str) -> Option<&Task> {
        self.tasks.iter().find(|task| task.id == id)
    }

    pub fn get_solution(&self, id: &str) -> Option<&Solution> {
        self.solutions
            .as_ref()
            .and_then(|sols| sols.iter().find(|s| s.id == id))
    }
}

impl Task {
    /// Get the train outputs
    #[inline]
    pub fn train_outputs(&self) -> Vec<&Grid> {
        self.train.iter().map(|example| &example.output).collect_vec()
    }

    /// Get the train inputs
    #[inline]
    pub fn train_inputs(&self) -> Vec<&Grid> {
        self.train.iter().map(|example| &example.input).collect_vec()
    }

    /// Get the test_inputs
    #[inline]
    pub fn test_inputs(&self) -> Vec<&Grid> {
        self.test.iter().map(|example| &example.input).collect_vec()
    }

    /// Get train inputs, train outputs, and test inputs
    #[inline]
    pub fn problem_grids(&self) -> Vec<&Grid> {
        [self.train_inputs(), self.train_outputs(), self.test_inputs()].concat()
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct TestSubmissionOutput {
    pub attempt_1: ARCGrid,
    pub attempt_2: ARCGrid,
}

pub type Submission = HashMap<String, Vec<TestSubmissionOutput>>;

impl Default for TestSubmissionOutput {
    fn default() -> Self {
        Self {
            attempt_1: vec![vec![0, 0], vec![0, 0]],
            attempt_2: vec![vec![0, 0], vec![0, 0]],
        }
    }
}
