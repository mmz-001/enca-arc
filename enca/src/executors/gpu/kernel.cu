static constexpr int NHBD_LEN = 5;
static constexpr int VIS_CHS = 4;
static constexpr int HID_CHS = 2;
static constexpr int INP_CHS = VIS_CHS * 2 + HID_CHS;
static constexpr int OUT_CHS = VIS_CHS + HID_CHS;
static constexpr int INP_DIM = NHBD_LEN * INP_CHS;
static constexpr int N_WEIGHTS = OUT_CHS * INP_DIM;
static constexpr int N_BIASES = OUT_CHS;
static constexpr int N_PARAMS = N_WEIGHTS + N_BIASES;
__device__ __constant__ static constexpr int NHBD[NHBD_LEN][2] = {
             { 0,-1},
    {-1, 0}, { 0, 0}, {1, 0},
             { 0, 1}
};

extern "C" __device__ void nca_update(float *__restrict__ sub, const int height, const int width,
                                      const float weights_t[INP_DIM][OUT_CHS], const float *__restrict__ biases) {
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
                outBuf[outCh] += (neighVal * mask) * weights_t[colIdx][outCh];
            }
        }
    }

    __syncthreads();

    // Update only writable channels
    for (int ch = 0; ch < OUT_CHS; ch++) {
        sub[base + ch + VIS_CHS] = __saturatef(outBuf[ch]);
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
    __shared__ float s_weights_t[INP_DIM][OUT_CHS];
    __shared__ float s_biases[N_BIASES];

    int grid_elem_base = (blockIdx.y * gridDim.x + blockIdx.x) * max_grid_size * INP_CHS;

    for (int i = 0; i < INP_CHS * size; i += size) {
        s_sub[threadIdx.x + i] = pop_subs[grid_elem_base + threadIdx.x + i];
    }

    for (int i = threadIdx.x; i < N_WEIGHTS; i += size) {
        int col = i % INP_DIM;
        int row = i / INP_DIM;
        s_weights_t[col][row] = pop_params[blockIdx.y * N_PARAMS + i];
    }

    for (int i = threadIdx.x; i < N_BIASES; i += size) {
        s_biases[i] = pop_params[blockIdx.y * N_PARAMS + N_WEIGHTS + i];
    }

    __syncthreads();

    for (int i = 0; i < max_steps; i++) {
        nca_update(s_sub, height, width, s_weights_t, s_biases);
        __syncthreads();
    }

    for (int i = 0; i < INP_CHS * size; i += size) {
        pop_subs[grid_elem_base + threadIdx.x + i] = s_sub[threadIdx.x + i];
    }
}
