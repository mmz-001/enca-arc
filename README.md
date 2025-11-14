# Evolved Neural Cellular Automata (ENCA) for ARC-AGI

Evolved Neural Cellular Automata (ENCA) solves ARC-AGI tasks with NCAs evolved using CMA-ES.

Blog: [enca-arc.mmzdev.com](https://enca-arc.mmzdev.com)

## Usage

Install Rust following instructions at [rustup.rs](https://rustup.rs/)

Linux setup

```shell
sudo apt update && sudo apt install git build-essential -y
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh && . "$HOME/.cargo/env"
git clone https://github.com/mmz-001/enca-arc
cd enca-arc/enca
```

Run all commands below in `/enca` package.

Unzip `enca/data.zip` containing ARC-AGI-1 and ARC-AGI-2 tasks:

```shell
unzip data.zip
```

## Training

Use the train binary to evolve NCAs.

Typical usage:
```shell
cargo run --release --bin train -- -t ./data/v1/arc-agi_training_challenges.json -a ./data/v1/arc-agi_training_solutions.json -r runs/v1-train -s 1 -c config.json
```

```
cargo run --release --bin train -- -t ./data/v2/arc-agi_evaluation_challenges.json -a ./data/v2/arc-agi_evaluation_solutions.json -r runs/v2-eval -s 1 -c config.json
```

Flags:
```
-h, --help                   Show help
-t, --tasks-path PATH        Path to tasks JSON
-a, --solutions-path PATH    Path to solutions JSON (used for evaluation)
-r, --out-dir DIR            Output directory for this run (default: runs/<timestamp>)
-i, --id TASK_ID             Optional single task id to train/evaluate
-s, --seed SEED              Optional seed for reproducibility.
-c, --config PATH            Optional path to config.json
```

## Visualization

> [!WARNING]
> **Photosensitive epilepsy warning**: The NCA visualizations can include rapid flashing and high-contrast transitions. Viewer discretion is advised.


After training, use the viz binary to interactively visualize execution.

Basic usage:
```shell
cargo run --release --bin viz -- -t ./data/v1/arc-agi_training_challenges.json -a ./data/v1/arc-agi_training_solutions.json -r runs/v1-train
```

```shell
cargo run --release --bin viz -- -t ./data/v2/arc-agi_evaluation_challenges.json -a ./data/v2/arc-agi_evaluation_solutions.json -r runs/v2-eval
```

Flags:
```
-h, --help                  Show help
-t, --tasks-path PATH       Tasks JSON file
-a, --solutions-path PATH   Solutions JSON file for evaluation
-r, --run-dir DIR           Run output directory (models and metrics)
-i, --id TASK_ID            Optional task id to open initially
```

Controls:
```
- Space: Pause/Resume simulation
- E: Toggle Train/Test split
- R: Reset current example
- D: Next example
- A: Previous example
- Shift+D / Mouse wheel up: Next task
- Shift+A / Mouse wheel down: Previous task
- Ctrl+C: Print task ID to console
```

## Running benchmarks

Run inference benchmarks with:

```shell
cargo bench --bench bench_nca -- --save-baseline nca
```

Compare against a saved baseline by
```shell
cargo bench --bench bench_nca -- --baseline nca
```

## Validation

The submission binary runs training and produces a `submission.json` file with:

```shell
cargo run --release --bin submission -- -p submission.json -a ./data/v1/arc-agi_training_solutions.json
```

Validate the submission using ground-truth solutions with:


```shell
cargo run --release --bin check -- -p submission.json -a ./data/v1/arc-agi_training_solutions.json
```

## Testing backends

The `gpu_check` binary runs random NCAs on both CPU and GPU backends and checks if the results are the same:

```shell
cargo run --release --bin gpu_check
```

## License

This project is dual-licensed under the MIT License and the Apache License 2.0; unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual-licensed as above without any additional terms or conditions.

## Contributing

Contributions are welcome!

## Acknowledgements

The Rust implementation of CMA-ES optimization algorithm is based on [cmaes](https://github.com/pengowen123/cmaes).
