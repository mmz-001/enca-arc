#pragma once

static constexpr int NHBD_LEN = 5;

static constexpr int NHBD[NHBD_LEN][2] = {
             { 0,-1},
    {-1, 0}, { 0, 0}, {1, 0},
             { 0, 1}
};

static constexpr int VIS_CHS = 4;
static constexpr int HID_CHS = 2;


static constexpr int RO_CH_START = 0;
static constexpr int RO_CH_END = VIS_CHS;              // 0..VIS_CHS

static constexpr int RW_CH_START = VIS_CHS;
static constexpr int RW_CH_END = 2 * VIS_CHS;          // VIS_CHS..(2*VIS_CHS)

static constexpr int INP_CHS = VIS_CHS * 2 + HID_CHS;  // 2*VIS + hid
static constexpr int HID_CH_START = 2 * VIS_CHS;
static constexpr int HID_CH_END = INP_CHS;             // (2*VIS_CHS)..INP_CHS

static constexpr int OUT_CHS = VIS_CHS + HID_CHS;

static constexpr int INP_DIM = NHBD_LEN * INP_CHS;

static constexpr int N_WEIGHTS = OUT_CHS * INP_DIM;
static constexpr int N_BIASES = OUT_CHS;

static constexpr int WEIGHTS_START = 0;
static constexpr int WEIGHTS_END = N_WEIGHTS;          // 0..N_WEIGHTS

static constexpr int BIASES_START = N_WEIGHTS;
static constexpr int BIASES_END = N_WEIGHTS + N_BIASES; // N_WEIGHTS..N_WEIGHTS+N_BIASES

static constexpr int N_PARAMS = N_WEIGHTS + N_BIASES;
static constexpr int MAX_GRID_SIZE = 30 * 30;
