# Evolved Neural Cellular Automata (ENCA) for ARC-AGI 

Evolved Neural Cellular Automata (ENCA) solves ARC-AGI problems using an ensemble of sequentially executed NCAs evolved using CMA-ES.

Blog: enca-arc.mmzdev.com

## Usage

Install Rust following instructions at [rustup.rs](https://rustup.rs/)

Linux setup

```shell
sudo apt update && sudo apt install git build-essential -y
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -y | sh && . "$HOME/.cargo/env"
git clone https://github.com/mmz-001/enca-arc
cd enca-arc
```

Download [ARC-AGI-1](https://www.kaggle.com/competitions/arc-prize-2024/data) and [ARC-AGI-2](https://www.kaggle.com/competitions/arc-prize-2025/data) datasets from Kaggle and put them under `./enca/data/v1` and `./enca/data/v2` respectively. Note the Kaggle format is different from the ARC-AGI-1 and ARC-AGI-2 repos.

## Training

Use the train binary to evolve NCA ensembles.


Typical usage:
```shell
cargo run --release --bin train -- -t ./data/v1/arc-agi_training_challenges.json -a ./data/v1/arc-agi_training_solutions.json -s 1
```


This saves the models, metrics, and summary in  `./runs/<timestamp>`

Flags:
```
-h, --help                   Show help
-t, --tasks-path PATH        Path to tasks JSON
-a, --solutions-path PATH    Path to solutions JSON (used for evaluation)
-r, --out-dir DIR            Output directory for this run (default: runs/<timestamp>)
-i, --id TASK_ID             Optional single task id to train/evaluate
-s, --seed SEED              Optional seed for reproducibility.
```

## Visualization

> [!WARNING]
> **Photosensitive epilepsy warning**: The NCA visualizations can include rapid flashing and high-contrast transitions. Viewer discretion is advised.


After training, use the viz binary to interactively visualize execution.

Basic usage:
```shell
cargo run --release --bin viz -- -t ./data/v1/arc-agi_training_challenges.json -a ./data/v1/arc-agi_training_solutions.json -r <run_dir>
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
- D: Next example (Shift+D: Next task)
- A: Previous example (Shift+A: Previous task)
```

## Running benchmarks

Put the model file `264363fd.json` in `enca/benches/models` and run inference benchmarks with:

```shell
cargo bench --bench bench_nca -- --save-baseline nca
```

Compare against a saved baseline by
```shell
cargo bench --bench bench_nca -- --baseline nca
```


## Static linking


```shell
sudo apt-get install musl-tools musl-dev -y
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl
```

Binaries under `target/x86_64-unknown-linux-musl/release`
If you're running the executable in a different machine ensure the target CPUs match since we set `target-cpu=native` in `RUSTFLAGS`. **Compiling for a generic CPU can cause severe performance degradation.**

## License

This project is dual-licensed under the MIT License and the Apache License 2.0; unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual-licensed as above without any additional terms or conditions.

## Contributing

Contributions are welcome!

## Acknowledgements

The Rust implementation of CMA-ES optimization algorithm is based on [cmaes](https://github.com/pengowen123/cmaes).
