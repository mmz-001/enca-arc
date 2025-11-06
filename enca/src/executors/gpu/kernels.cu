#include "constants.h"

extern "C" __device__ void nca_update(float *__restrict__ sub, const int height, const int width,
                                      const float *__restrict__ weights, const float *__restrict__ biases) {
    const int x = threadIdx.x % width;
    const int y = threadIdx.x / width;
    int base = threadIdx.x * INP_CHS;

    float outBuf[OUT_CHS];

    for (int i = 0; i < OUT_CHS; i++) {
        outBuf[i] = biases[i];
    }

    #pragma unroll
    for (int ni = 0; ni < NHBD_LEN; ni++) {
        const int nx = x + NHBD[ni][0];
        const int ny = y + NHBD[ni][1];

        if ((unsigned)nx >= (unsigned)width || (unsigned)ny >= (unsigned)height) {
            continue;
        }

        const int nbase = (ny * width + nx) * INP_CHS;

        #pragma unroll
        for (int inCh = 0; inCh < INP_CHS; inCh++) {

            const float neighVal = sub[nbase + inCh];
            // Alive masking
            const float mask = (neighVal >= 0.5f) ? 1.0f : 0.0f;
            const int colIdx = inCh * NHBD_LEN + ni;

            for (int outCh = 0; outCh < OUT_CHS; outCh++) {
                const int wi = outCh * INP_DIM + colIdx;
                outBuf[outCh] += (neighVal * mask) * weights[wi];
            }
        }
    }

    __syncthreads();

    // Update only writable channels
    for (int ch = 0; ch < OUT_CHS; ch++) {
        sub[base + ch + VIS_CHS] = __saturatef(outBuf[ch]);
    }
}

extern "C" __device__ void nca_executor_run(float *__restrict__ sub, const float *__restrict__ params,
                                            const int height, const int width, const int max_steps) {

    const float *s_weights = params;
    const float *s_biases = &params[N_WEIGHTS];

    for (int i = 0; i < max_steps; i++) {
        nca_update(sub, height, width, s_weights, s_biases);
        __syncthreads();
    }
}

extern "C" __global__ void pop_nca_executor_run_batch(float *__restrict__ pop_subs,
                                                      const float *__restrict__ pop_params,
                                                      const int *__restrict__ heights, const int *__restrict__ widths,
                                                      const int max_steps, const int max_grid_size) {
    int height = heights[blockIdx.x];
    int width = widths[blockIdx.x];
    int size = height * width;

    if (threadIdx.x >= size) {
        return;
    }

    extern __shared__ float s_sub[];
    __shared__ float s_params[N_PARAMS];

    int base = threadIdx.x * INP_CHS;
    int n_grids = gridDim.x;
    int ind_idx = blockIdx.y;
    int grid_idx = blockIdx.x;

    int grid_elem_base = (ind_idx * n_grids + grid_idx) * max_grid_size;
    int subs_base = (grid_elem_base + threadIdx.x) * INP_CHS;

    for (int ch = 0; ch < INP_CHS; ch++) {
        s_sub[base + ch] = pop_subs[subs_base + ch];
    }

    for (int i = threadIdx.x; i < N_PARAMS; i += size) {
        s_params[i] = pop_params[ind_idx * N_PARAMS + i];
    }

    __syncthreads();

    nca_executor_run(s_sub, s_params, height, width, max_steps);

    for (int ch = 0; ch < INP_CHS; ch++) {
        pop_subs[subs_base + ch] = s_sub[base + ch];
    }
}
