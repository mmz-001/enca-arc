# Changelog

All notable changes to the architecture, training, and inference will be documented in this file.

## [2.0.0] - 2025-11-06

This release introduces experimental multi-GPU inference and a simplified architecture.

- Switch to a single-NCA architecture. A single NCA is powerful enough to solve a lot of ARC-AGI problems when properly trained. See [experiment logs](https://github.com/mmz-001/enca-arc-experiments) for performance changes. The code is much simpler and generalizes better to problems outside ARC-AGI.

- Increase the number of hidden channels to 2. The model now has 306 parameters total, which is approximately 70% fewer than V1.

- Add experimental multi-GPU support with custom CUDA kernels for population-based training. Each generation of CMA-ES now runs entirely on the GPU(s). This is an initial draft and there is substantial room for kernel optimizations.

- Remove convergence checks. This simplifies the CUDA code at the expense of making the CPU code somewhat slower.

- Simplify fitness function. It now uses only the mean pixel-wise MSE and L2 weight regularization. The L1, oscillation, and non-convergence penalties were removed.

- Introduce sparse parameter updates. Full CMA-ES covariance evaluation is computationally infeasible at higher parameter counts, so it is now computed on random subsets of parameters. The training code was overhauled to address training dynamics introduced by sparse updates. See `/enca/src/solver.rs` for implementation details.

## [1.0.0] - 2025-10-09

ENCA V1 uses a multi-NCA architecture that sequentially executes NCAs on the same substrate. Details can be found in my [blog post](https://enca-arc.mmzdev.com/).
