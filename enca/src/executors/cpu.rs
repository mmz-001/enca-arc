use crate::{
    constants::{INP_CHS, NHBD, OUT_CHS, VIS_CHS},
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
                        let row_idx = ni * INP_CHS + inp_ch_idx;
                        let neighbor_val = data[(ny as usize, nx as usize, inp_ch_idx)];

                        // Alive masking
                        if neighbor_val < 0.5 {
                            continue;
                        }

                        for i in 0..OUT_CHS {
                            let wi = row_idx * OUT_CHS + i;
                            out_buf[i] = f32::mul_add(neighbor_val, self.nca.weights[wi], out_buf[i]);
                        }
                    }
                }

                // Update only writable channels.
                for ch in 0..OUT_CHS {
                    next[(y, x, ch + VIS_CHS)] = (next[(y, x, ch + VIS_CHS)] + out_buf[ch]).clamp(0.0, 1.0);
                }
            }
        }

        substrate.data = next;
    }
}
