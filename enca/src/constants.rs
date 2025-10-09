// Von-Neumann neighborhood
#[rustfmt::skip]
pub const NHBD: [(i32, i32); 5] = [
              ( 0,-1),
     (-1, 0), ( 0, 0), (1, 0),
              ( 0, 1)
];

// Moore neighborhood
// #[rustfmt::skip]
// pub const NHBD: [(i32, i32); 9] = [
//      (-1,-1), ( 0,-1), (1,-1),
//      (-1, 0), ( 0, 0), (1, 0),
//      (-1, 1), ( 0, 1), (1, 1),
// ];

pub const NHBD_LEN: usize = NHBD.len();

// Index of the neighborhood center
pub const NHBD_CENTER: usize = NHBD_LEN / 2;

/// Number of visible channels (RO or RW)
pub const VIS_CHS: usize = 4;
/// Number of hidden channels
pub const HID_CHS: usize = 1;

/// Identity color map
pub const I_COL_MAP: [u8; 10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

/// Max inference time color permutations
pub const MAX_PERMUTATIONS: usize = 1000;

/// Index range of read/write visible channels
pub const RW_CH_RNG: std::ops::Range<usize> = VIS_CHS..(2 * VIS_CHS);

/// Index range of read only visible channels
pub const RO_CH_RNG: std::ops::Range<usize> = 0..VIS_CHS;

/// Number of input channels
pub const INP_CHS: usize = VIS_CHS * 2 + HID_CHS;

/// Index range of hidden channels
pub const HID_CH_RNG: std::ops::Range<usize> = (2 * VIS_CHS)..(INP_CHS);

pub const OUT_CHS: usize = VIS_CHS + HID_CHS;

/// Input dimensions of NCA
pub const INP_DIM: usize = NHBD_LEN * INP_CHS;
