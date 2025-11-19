static constexpr int NHBD_LEN = 5;
static constexpr int VIS_CHS = 4;
static constexpr int HID_CHS = 2;
static constexpr int INP_CHS = VIS_CHS * 2 + HID_CHS;
static constexpr int OUT_CHS = VIS_CHS + HID_CHS;
static constexpr int INP_DIM = NHBD_LEN * INP_CHS;
static constexpr int N_WEIGHTS = INP_DIM * OUT_CHS;
static constexpr int N_BIASES = OUT_CHS;
static constexpr int N_PARAMS = N_WEIGHTS + N_BIASES;
__device__ __constant__ static constexpr int NHBD[NHBD_LEN][2] = {
             { 0,-1},
    {-1, 0}, { 0, 0}, {1, 0},
             { 0, 1}
};

extern "C" __device__ void nca_update(float *__restrict__ sub, const int height, const int width,
                                      const float weights[N_PARAMS], const float *__restrict__ biases) {
    const int x = threadIdx.x % width;
    const int y = threadIdx.x / width;
    int base = threadIdx.x * INP_CHS;

    float out_buf[OUT_CHS];

    for (int i = 0; i < OUT_CHS; i++) {
        out_buf[i] = biases[i];
    }

    #pragma unroll
    for (int ni = 0; ni < NHBD_LEN; ni++) {
        const int nx = x + NHBD[ni][0];
        const int ny = y + NHBD[ni][1];

        if ((unsigned)nx >= (unsigned)width || (unsigned)ny >= (unsigned)height) {
            continue;
        }

        const int n_base = (ny * width + nx) * INP_CHS;

        #pragma unroll
        for (int inCh = 0; inCh < INP_CHS; inCh++) {

            const float neigh_val = sub[n_base + inCh];

            // Alive masking
            if (neigh_val < 0.5f) {
                continue;
            }

            const int col_idx = inCh * NHBD_LEN + ni;
            for (int out_ch = 0; out_ch < OUT_CHS; out_ch++) {
                int wi = col_idx * OUT_CHS + out_ch;
                out_buf[out_ch] += neigh_val * weights[wi];
            }
        }
    }

    __syncthreads();

    // Update only writable channels
    for (int ch = 0; ch < OUT_CHS; ch++) {
        sub[base + ch + VIS_CHS] = __saturatef(out_buf[ch] + sub[base + ch + VIS_CHS]);
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

    extern __shared__ float sub_s[];
    __shared__ float weights_s[N_PARAMS];
    __shared__ float biases_s[N_BIASES];

    int grid_elem_base = (blockIdx.y * gridDim.x + blockIdx.x) * max_grid_size * INP_CHS;

    for (int i = 0; i < INP_CHS * size; i += size) {
        sub_s[threadIdx.x + i] = pop_subs[grid_elem_base + threadIdx.x + i];
    }

    for (int i = threadIdx.x; i < N_WEIGHTS; i += size) {
        weights_s[i] = pop_params[blockIdx.y * N_PARAMS + i];
    }

    for (int i = threadIdx.x; i < N_BIASES; i += size) {
        biases_s[i] = pop_params[blockIdx.y * N_PARAMS + N_WEIGHTS + i];
    }

    __syncthreads();

    for (int i = 0; i < max_steps; i++) {
        nca_update(sub_s, height, width, weights_s, biases_s);
        __syncthreads();
    }

    for (int i = 0; i < INP_CHS * size; i += size) {
        pop_subs[grid_elem_base + threadIdx.x + i] = sub_s[threadIdx.x + i];
    }
}
