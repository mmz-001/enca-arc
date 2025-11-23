use crate::{
    config::Config,
    constants::{INP_DIM, N_BIASES, N_WEIGHTS, OUT_CHS},
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
    pub vis_steps: usize,
    pub hid_steps: usize,
    pub transform_pipeline: TransformPipeline,
}

impl NCA {
    pub fn new(config: Config) -> Self {
        let weights = vec![0.0; INP_DIM * OUT_CHS];
        let biases = vec![0.0; OUT_CHS];

        Self {
            weights,
            biases,
            vis_steps: config.vis_steps,
            hid_steps: config.hid_steps,
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

    pub fn from_vec(weights: &[f32], biases: &[f32], config: Config) -> Self {
        let mut nca = Self::new(config);

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
