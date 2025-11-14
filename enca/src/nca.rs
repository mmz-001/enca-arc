use crate::{
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
