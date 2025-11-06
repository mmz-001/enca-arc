use clap::Parser;
use enca::{
    augment::TaskNCAs,
    dataset::{Dataset, Solution, Task},
    drawing::{display_visible_grid, draw_metrics, draw_params, draw_tooltip},
    executors::{Backend, NCAExecutor},
    grid::Grid,
    metrics::TaskReport,
    serde_utils::JSONReadWrite,
};
use macroquad::Window;
use macroquad::{prelude::*, window::Conf};
use std::thread;

#[derive(Parser, Debug)]
struct Args {
    /// Tasks JSON file
    #[arg(short = 't', long)]
    tasks_path: String,
    /// Solutions JSON file for evaluation
    #[arg(short = 'a', long)]
    solutions_path: String,
    // Run output directory
    #[arg(short = 'r', long)]
    run_dir: String,
    // Task id
    #[arg(short = 'i', long)]
    id: Option<String>,
}

#[derive(Copy, Clone)]
enum Action {
    Reset,
    NextExample,
    PrevExample,
    NextTask,
    PrevTask,
    ToggleSplit,
    TogglePause,
    ToggleHelp,
}

#[derive(Clone, Copy)]
struct Layout {
    gx: f32,
    gy: f32,
    gs: f32,
    cs: f32,
    ig_pad: f32,
    igx: f32,
    igy: f32,
    params_x: f32,
    params_y: f32,
    params_w: f32,
    params_h: f32,
    metrics_x: f32,
    metrics_y: f32,
    metrics_w: f32,
    metrics_h: f32,
}

fn compute_layout(sw: f32, sh: f32) -> Layout {
    let ss = sw.min(sh);
    let gs = ss / 2.0;
    let gx = gs / 2.0;
    let gy = gs / 3.0;
    let cs = gs / 5.0;
    let ig_pad = cs * 1.5;
    let igx = gx + gs + ig_pad;
    let igy = gy;
    let params_x = gx;
    let params_y = gy + gs + 100.0;
    let params_w = gs;
    let params_h = (sh - gs) / 8.0;

    let metrics_x = igx;
    let metrics_y = params_y;
    let metrics_w = gs * 1.5;
    let metrics_h = (sh - gs) / 2.5;

    Layout {
        gx,
        gy,
        gs,
        cs,
        ig_pad,
        igx,
        igy,
        params_x,
        params_y,
        params_w,
        params_h,
        metrics_x,
        metrics_y,
        metrics_w,
        metrics_h,
    }
}

struct AppState {
    // Data
    dataset: Dataset,
    task_ncas: Vec<(String, TaskNCAs)>,
    metrics: Vec<(String, TaskReport)>,

    // Selection
    current_task: Task,
    current_solution: Solution,
    current_task_idx: usize,
    split: Split,
    example_id: usize,

    // Runtime
    input: Grid,
    output: Grid,
    executor: NCAExecutor,
    paused: bool,
    show_help: bool,

    // Timing
    fps: f64,
    acc: f64,
}

impl AppState {
    fn new(
        dataset: Dataset,
        task_ncas: Vec<(String, TaskNCAs)>,
        metrics: Vec<(String, TaskReport)>,
        initial_task_idx: usize,
        fps: f64,
    ) -> Self {
        let current_task_idx = initial_task_idx.min(task_ncas.len().saturating_sub(1));
        let current_task_id = task_ncas[current_task_idx].0.clone();

        let current_task = dataset
            .get_task(&current_task_id)
            .unwrap_or_else(|| panic!("task_id={} not found", current_task_id))
            .clone();

        let current_solution = dataset
            .get_solution(&current_task_id)
            .unwrap_or_else(|| panic!("task_id={} not found", current_task_id))
            .clone();

        let example_id = 0usize;
        let example = current_task.train[example_id].clone();

        let (input, output) = (&example.input, &example.output);

        let executor = NCAExecutor::new(task_ncas[current_task_idx].1.train.clone(), input, Backend::CPU);

        AppState {
            dataset,
            task_ncas,
            metrics,
            current_task_idx,
            current_task,
            current_solution,
            split: Split::Train,
            example_id,
            input: input.clone(),
            output: output.clone(),
            executor,
            paused: false,
            show_help: false,
            fps,
            acc: 0.0,
        }
    }

    fn handle_input(&self) -> Vec<Action> {
        let mut actions = Vec::new();
        let shift_down = is_key_down(KeyCode::LeftShift) || is_key_down(KeyCode::RightShift);

        if is_key_pressed(KeyCode::Space) {
            actions.push(Action::TogglePause);
        }
        if is_key_pressed(KeyCode::H) {
            actions.push(Action::ToggleHelp);
        }
        if is_key_pressed(KeyCode::E) {
            actions.push(Action::ToggleSplit);
        }
        if is_key_pressed(KeyCode::R) {
            actions.push(Action::Reset);
        }

        if is_key_pressed(KeyCode::D) {
            actions.push(if shift_down {
                Action::NextTask
            } else {
                Action::NextExample
            });
        }
        if is_key_pressed(KeyCode::A) {
            actions.push(if shift_down {
                Action::PrevTask
            } else {
                Action::PrevExample
            });
        }

        if mouse_wheel().1 > 0.0 {
            actions.push(Action::NextTask)
        }

        if mouse_wheel().1 < 0.0 {
            actions.push(Action::PrevTask)
        }

        if is_key_pressed(KeyCode::C) && is_key_down(KeyCode::LeftControl) {
            println!("task_id={}", self.current_task.id);
        }

        actions
    }

    // Apply actions and trigger a single rebuild if needed
    fn process_actions(&mut self, actions: &[Action]) {
        let mut rebuild_needed = false;

        for &a in actions {
            match a {
                Action::TogglePause => {
                    self.paused = !self.paused;
                }
                Action::ToggleHelp => {
                    self.show_help = !self.show_help;
                }
                Action::ToggleSplit => {
                    self.split = match self.split {
                        Split::Train => Split::Test,
                        Split::Test => Split::Train,
                    };
                    self.example_id = 0;
                    rebuild_needed = true;
                }
                Action::Reset => {
                    rebuild_needed = true;
                }
                Action::NextExample => {
                    let num_examples = match self.split {
                        Split::Train => self.current_task.train.len(),
                        Split::Test => self.current_task.test.len(),
                    };
                    self.example_id = (self.example_id + 1).min(num_examples - 1);
                    rebuild_needed = true;
                }
                Action::PrevExample => {
                    self.example_id = self.example_id.saturating_sub(1);
                    rebuild_needed = true;
                }
                Action::NextTask => {
                    self.current_task_idx = (self.current_task_idx + 1) % self.task_ncas.len();
                    self.current_task = self
                        .dataset
                        .get_task(&self.task_ncas[self.current_task_idx].0)
                        .unwrap_or_else(|| panic!("task_id={} not found", self.task_ncas[self.current_task_idx].0))
                        .clone();
                    self.current_solution = self
                        .dataset
                        .get_solution(&self.current_task.id)
                        .unwrap_or_else(|| panic!("task_id={} not found", &self.current_task.id))
                        .clone();
                    self.split = Split::Train;
                    self.example_id = 0;
                    rebuild_needed = true;
                }
                Action::PrevTask => {
                    self.current_task_idx = (self.task_ncas.len() + self.current_task_idx - 1) % self.task_ncas.len();
                    self.current_task = self
                        .dataset
                        .get_task(&self.task_ncas[self.current_task_idx].0)
                        .unwrap_or_else(|| panic!("task_id={} not found", self.task_ncas[self.current_task_idx].0))
                        .clone();
                    self.current_solution = self
                        .dataset
                        .get_solution(&self.current_task.id)
                        .unwrap_or_else(|| panic!("task_id={} not found", &self.current_task.id))
                        .clone();
                    self.split = Split::Train;
                    self.example_id = 0;
                    rebuild_needed = true;
                }
            }
        }

        if rebuild_needed {
            self.rebuild_context();
        }
    }

    fn rebuild_context(&mut self) {
        self.current_task = self
            .dataset
            .get_task(&self.task_ncas[self.current_task_idx].0)
            .unwrap_or_else(|| panic!("task_id={} missing", self.current_task.id))
            .clone();
        self.current_solution = self
            .dataset
            .get_solution(&self.current_task.id)
            .unwrap_or_else(|| panic!("task_id={} missing", &self.current_task.id))
            .clone();

        match self.split {
            Split::Train => {
                let examples = &self.current_task.train;
                let n_examples = examples.len();
                assert!(n_examples > 0, "Task '{}' has no train examples", self.current_task.id);

                // Normalize example_id into range
                self.example_id %= n_examples;

                // Rebuild input/output
                self.input = examples[self.example_id].input.clone();
                self.output = examples[self.example_id].output.clone();
            }
            Split::Test => {
                let examples = &self.current_task.test;
                let n_examples = examples.len();
                assert!(n_examples > 0, "Task '{}' has no test examples", self.current_task.id);

                // Normalize example_id into range
                self.example_id %= n_examples;

                // Rebuild input/output
                self.input = examples[self.example_id].input.clone();
                self.output = self.current_solution.outputs[self.example_id].clone();
            }
        }

        let mut nca = self.task_ncas[self.current_task_idx].1.train.clone();

        if self.split == Split::Test {
            // Use augmented version
            nca = self.task_ncas[self.current_task_idx].1.test[self.example_id].clone();
        }

        self.executor = NCAExecutor::new(nca, &self.input, Backend::CPU);

        // Reset sim timing flags
        self.paused = false;
        self.acc = 0.0;
    }

    fn step_sim(&mut self) {
        if self.paused {
            return;
        }
        self.acc += get_frame_time() as f64;
        let step_dt = 1.0 / self.fps;

        while self.acc >= step_dt && !self.paused {
            self.acc -= step_dt;
            self.paused = self.executor.step();
        }
    }

    fn draw(&mut self) {
        clear_background(BLACK);

        let sw = screen_width();
        let sh = screen_height();
        let l = compute_layout(sw, sh);

        // Work with the currently selected executor
        let substrate = &self.executor.substrate();

        // Main grids
        let mut grid = substrate.to_grid();
        let input = self.input.clone();
        let output = self.output.clone();

        let transforms = &self.executor.nca().transform_pipeline;
        transforms.revert(&mut grid);

        display_visible_grid(&grid, l.gx, l.gy, l.gs, l.gs);
        display_visible_grid(&input, l.igx, l.igy, l.cs, l.cs);
        display_visible_grid(&output, l.igx + l.ig_pad, l.gy, l.cs, l.cs);

        substrate.display_channels_panel(l.igx, l.igy + l.ig_pad, l.gs * 1.1, l.gs / 1.4);

        // Params UI
        draw_params(l.params_x, l.params_y, l.params_w, l.params_h, &self.executor.nca());

        let (clicked, tooltip) = draw_metrics(
            l.metrics_x,
            l.metrics_y,
            l.metrics_w,
            l.metrics_h,
            &self.dataset,
            &self.metrics,
            Some(self.current_task.id.as_str()),
        );

        if let Some(task_id) = clicked
            && let Some(idx) = self.task_ncas.iter().position(|(id, _)| id == &task_id)
            && idx != self.current_task_idx
        {
            {
                self.current_task_idx = idx;
                self.split = Split::Train;
                self.example_id = 0;
                self.rebuild_context();
            }
        }

        if let Some((tx, ty, lines_owned)) = tooltip {
            let lines: Vec<&str> = lines_owned.iter().map(|s| s.as_str()).collect();
            draw_tooltip(tx, ty, &lines);
        }

        // Status
        let status = format!(
            "task_id={}, example_id={}, split={}, steps={}/{}",
            &self.current_task.id,
            self.example_id,
            self.split,
            self.executor.steps(),
            self.executor.nca().max_steps
        );
        draw_text(&status, l.gx, l.gy - 4.0, 24.0, WHITE);
    }
}

async fn draw(
    dataset: Dataset,
    augmented_ncas: Vec<(String, TaskNCAs)>,
    metrics: Vec<(String, TaskReport)>,
    id: Option<String>,
) {
    let initial_task_idx = if let Some(ref id) = id {
        augmented_ncas.iter().position(|x| &x.0 == id).unwrap_or(0)
    } else {
        0
    };

    let mut app = AppState::new(
        dataset,
        augmented_ncas,
        metrics,
        initial_task_idx,
        10.0, // fps
    );

    loop {
        let actions = app.handle_input();
        app.process_actions(&actions);

        app.step_sim();
        app.draw();

        next_frame().await;
    }
}

fn main() {
    let args = Args::parse();
    let tasks_path = args.tasks_path;
    let solutions_path = args.solutions_path;
    let run_dir = args.run_dir;
    let models_dir = format!("{run_dir}/models");
    let metrics_dir = format!("{run_dir}/metrics");

    let dataset = Dataset::load(&tasks_path, Some(&solutions_path));

    let augmented_ncas = TaskNCAs::load(&models_dir).unwrap_or_else(|e| {
        panic!("Failed to read run output models directory '{}': {}", models_dir, e);
    });

    let metrics = TaskReport::load(&metrics_dir).unwrap_or_else(|e| {
        panic!("Failed to read run output models directory '{}': {}", metrics_dir, e);
    });

    if augmented_ncas.is_empty() {
        panic!("No models found in {models_dir}");
    }

    if metrics.is_empty() {
        panic!("No metrics found in {metrics_dir}");
    }

    thread::spawn(|| {
        Window::from_config(
            Conf {
                window_title: "ENCA".to_owned(),
                fullscreen: true,
                high_dpi: true,
                sample_count: 16,
                ..Default::default()
            },
            draw(dataset, augmented_ncas, metrics, args.id),
        );
    })
    .join()
    .unwrap();
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Split {
    Train,
    Test,
}

impl std::fmt::Display for Split {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = match self {
            Split::Train => "train",
            Split::Test => "test",
        };
        write!(f, "{str}")
    }
}
