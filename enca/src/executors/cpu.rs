use crate::{
    constants::{INP_CHS, INP_DIM, NHBD, NHBD_LEN, OUT_CHS, VIS_CHS},
    grid::Grid,
    nca::NCA,
    substrate::Substrate,
};

/// Handles NCA step updates and stores execution state
#[derive(Clone)]
pub struct NCAExecutorCpu {
    pub nca: NCA,
    pub steps: usize,
    pub substrate: Substrate,
}

impl NCAExecutorCpu {
    pub fn new(nca: NCA, grid: &Grid) -> Self {
        let mut grid = grid.clone();

        nca.transform_pipeline.apply(&mut grid);
        let substrate = Substrate::from_grid(&grid);

        Self {
            nca,
            steps: 0,
            substrate,
        }
    }

    pub fn run(&mut self) {
        for _ in 0..self.nca.max_steps {
            self.step();
        }
    }

    /// Executes one iteration step.
    pub fn step(&mut self) -> bool {
        if self.steps >= self.nca.max_steps {
            return true;
        }
        self.update();
        self.steps += 1;
        false
    }

    /// Updates the substrate using the NCA
    pub fn update(&mut self) {
        let substrate = &mut self.substrate;
        let mut next = substrate.data.clone();

        let w = substrate.width as i32;
        let h = substrate.height as i32;

        let data = substrate.data.view();
        let mut out_buf = [0.0; OUT_CHS];

        for y in 0..substrate.height {
            for x in 0..substrate.width {
                for i in 0..OUT_CHS {
                    out_buf[i] = self.nca.biases[i]
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
                                f32::mul_add(neighbor_val, unsafe { *self.nca.weights.get_unchecked(wi) }, out_buf[i]);
                        }
                    }
                }

                // Update only writable channels.
                for ch in 0..OUT_CHS {
                    *unsafe { next.get_mut((y, x, ch + VIS_CHS)).unwrap_unchecked() } += out_buf[ch]
                }
            }
        }

        next.mapv_inplace(|v| v.clamp(0.0, 1.0));

        substrate.data = next;
    }
}
