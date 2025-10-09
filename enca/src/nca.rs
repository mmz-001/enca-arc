use crate::{
    constants::{INP_CHS, INP_DIM, NHBD, NHBD_CENTER, NHBD_LEN, OUT_CHS, VIS_CHS},
    substrate::Substrate,
    transforms::TransformPipeline,
};
use mimalloc::MiMalloc;
use serde::{Deserialize, Serialize};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Serialize, Deserialize, Clone)]
pub struct NCA {
    pub weights: Vec<f32>,
    pub biases: Vec<f32>,
    pub max_steps: usize,
}

impl NCA {
    pub fn new(max_steps: usize) -> Self {
        let weights = vec![0.0; OUT_CHS * INP_DIM];
        let biases = vec![0.0; OUT_CHS];

        let mut nca = Self {
            weights,
            biases,
            max_steps,
        };

        nca.initialize_identity();

        nca
    }

    /// Sets all weights and biases to zero
    pub fn clear(&mut self) {
        self.weights.fill(0.0);
        self.biases.fill(0.0);
    }

    /// Initialize this NCA to pass through RW channels (copies RW input -> RW output).
    /// Leaves hidden channels untouched.
    pub fn initialize_identity(&mut self) {
        // Copy RW visible channels: input RW bit -> output RW bit at center tap
        for bit in 0..VIS_CHS {
            let ch_in = VIS_CHS + bit; // RW input channel index in substrate
            let col_idx = ch_in * NHBD_LEN + NHBD_CENTER;
            // Row 'bit' corresponds to the RW output channel 'bit'
            self.weights[bit * INP_DIM + col_idx] = 1.0;
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
                        let neighbor_val = data[(ny as usize, nx as usize, inp_ch_idx)];

                        // Alive masking
                        if neighbor_val < 0.5 {
                            continue;
                        }

                        let col_idx = inp_ch_idx * NHBD_LEN + ni;

                        for i in 0..OUT_CHS {
                            let wi = i * INP_DIM + col_idx;
                            out_buf[i] = f32::mul_add(neighbor_val, self.weights[wi], out_buf[i]);
                        }
                    }
                }

                // Update only writable channels.
                for ch in 0..OUT_CHS {
                    next[(y, x, ch + VIS_CHS)] = out_buf[ch]
                }
            }
        }

        next.mapv_inplace(|v| v.clamp(0.0, 1.0));

        substrate.data = next;
    }

    pub fn from_vec(params: Vec<f32>, max_steps: usize) -> Self {
        let mut nca = Self::new(max_steps);

        let weights_len = OUT_CHS * INP_DIM;
        let biases_len = OUT_CHS;
        let expected_total = weights_len + biases_len;
        if params.len() != expected_total {
            panic!(
                "Expected {} total params ({} weights + {} biases); found {}",
                expected_total,
                weights_len,
                biases_len,
                params.len()
            );
        }

        nca.weights = params[..weights_len].to_vec();
        nca.biases = params[weights_len..].to_vec();

        nca
    }

    pub fn to_vec(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.weights.len() + self.biases.len());
        out.extend(self.weights.iter().map(|&x| x as f64));
        out.extend(self.biases.iter().map(|&x| x as f64));
        out
    }
}

/// Collection of NCA for a given task along with grid
/// transformations applied during inference
#[derive(Serialize, Deserialize, Clone)]
pub struct NCAEnsemble {
    pub task_id: String,
    pub ncas: Vec<NCA>,
    pub transform_pipeline: TransformPipeline,
}

impl NCAEnsemble {
    pub fn new(ncas: Vec<NCA>, task_id: String) -> Self {
        Self {
            task_id,
            ncas,
            transform_pipeline: TransformPipeline { steps: vec![] },
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{constants::RW_CH_RNG, executors::NCAExecutor};
    use ndarray::{Array3, s};
    use ndarray_rand::{RandomExt, rand_distr::Uniform};

    #[test]
    fn initialize_identity_no_change() {
        let width = 10;
        let height = 10;

        let mut data = Array3::<f32>::random((height, width, INP_CHS), Uniform::new(0.0, 1.0));
        data.mapv_inplace(|v| v.clamp(0.5, 1.0));

        let substrate = Substrate { data, width, height };

        let nca = NCA::new(30);

        let mut executor = NCAExecutor::new(nca, substrate.clone());

        executor.run();

        println!("steps={}", executor.steps);

        let rw_channels = s![.., .., RW_CH_RNG];
        let diff = (&substrate.data.slice(rw_channels) - &executor.substrate.data.slice(rw_channels))
            .abs()
            .mean()
            .unwrap();
        assert!(diff <= 1e-12, "Substrate changed. diff={:.3e}", diff);
    }
}
