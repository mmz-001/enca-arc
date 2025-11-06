use crate::{
    constants::{INP_CHS, INP_DIM, N_BIASES, N_WEIGHTS, NHBD, NHBD_LEN, OUT_CHS, VIS_CHS},
    substrate::Substrate,
    transforms::TransformPipeline,
};
use mimalloc::MiMalloc;
use rand::Rng;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Serialize, Deserialize, Clone)]
pub struct NCA {
    pub weights: Vec<f32>,
    pub biases: Vec<f32>,
    pub max_steps: usize,
    pub transform_pipeline: TransformPipeline,
}

impl NCA {
    pub fn new(max_steps: usize) -> Self {
        let weights = vec![0.0; OUT_CHS * INP_DIM];
        let biases = vec![0.0; OUT_CHS];

        Self {
            weights,
            biases,
            max_steps,
            transform_pipeline: TransformPipeline::default(),
        }
    }

    /// Initialize weights and biases with small random values
    pub fn initialize_random(&mut self, rng: &mut impl Rng) {
        let dist = Normal::new(0.0, 0.2).unwrap();

        for weight in self.weights.iter_mut() {
            *weight = rng.sample(dist);
        }

        for bias in self.biases.iter_mut() {
            *bias = rng.sample(dist);
        }
    }

    /// Updates the substrate using the NCA
    pub fn update(&self, substrate: &mut Substrate) {
        let mut next = substrate.data.clone();

        let w = substrate.width as i32;
        let h = substrate.height as i32;

        let data = substrate.data.view();
        let mut out_buf = [0.0; OUT_CHS];

        for y in 0..substrate.height {
            for x in 0..substrate.width {
                for i in 0..OUT_CHS {
                    out_buf[i] = self.biases[i]
                }

                for (ni, (dx, dy)) in NHBD.iter().enumerate() {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || nx >= w || ny < 0 || ny >= h {
                        // Out of bounds
                        continue;
                    };

                    for inp_ch_idx in 0..INP_CHS {
                        let neighbor_val =
                            unsafe { *data.get((ny as usize, nx as usize, inp_ch_idx)).unwrap_unchecked() };

                        // Alive masking
                        if neighbor_val < 0.5 {
                            continue;
                        }

                        let col_idx = inp_ch_idx * NHBD_LEN + ni;

                        for i in 0..OUT_CHS {
                            let wi = i * INP_DIM + col_idx;
                            out_buf[i] =
                                f32::mul_add(neighbor_val, unsafe { *self.weights.get_unchecked(wi) }, out_buf[i]);
                        }
                    }
                }

                // Update only writable channels.
                for ch in 0..OUT_CHS {
                    *unsafe { next.get_mut((y, x, ch + VIS_CHS)).unwrap_unchecked() } = out_buf[ch]
                }
            }
        }

        next.mapv_inplace(|v| v.clamp(0.0, 1.0));

        substrate.data = next;
    }

    pub fn from_vec(weights: &[f32], biases: &[f32], max_steps: usize) -> Self {
        let mut nca = Self::new(max_steps);

        if weights.len() != N_WEIGHTS {
            panic!("Expected {} weights; found {}", N_WEIGHTS, weights.len())
        }

        if biases.len() != N_BIASES {
            panic!("Expected {} biases; found {}", N_BIASES, biases.len());
        }

        nca.weights = weights.to_vec();
        nca.biases = biases.to_vec();

        nca
    }

    pub fn to_vec(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.weights.len() + self.biases.len());
        out.extend(self.weights.to_vec());
        out.extend(self.biases.to_vec());
        out
    }
}
