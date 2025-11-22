use crate::{
    constants::{HID_CHS, INP_CHS, NHBD, OUT_CHS, VIS_CHS},
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
        self.update_hidden();
        self.update_rw();
        self.steps += 1;
        false
    }

    // Updates hidden channels
    pub fn update_hidden(&mut self) {
        let substrate = &mut self.substrate;
        let mut next = substrate.data.clone();

        let w = substrate.width as i32;
        let h = substrate.height as i32;

        let data = substrate.data.view();
        let mut out_buf = [0.0; HID_CHS];

        for y in 0..substrate.height {
            for x in 0..substrate.width {
                for i in 0..HID_CHS {
                    out_buf[i] = self.nca.biases[VIS_CHS + i]
                }

                for (ni, (dx, dy)) in NHBD.iter().enumerate() {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || nx >= w || ny < 0 || ny >= h {
                        // Out of bounds
                        continue;
                    };

                    for ch_idx in 0..INP_CHS {
                        let row_idx = ni * INP_CHS + ch_idx;
                        let neighbor_val = data[(ny as usize, nx as usize, ch_idx)];

                        // Alive masking
                        if neighbor_val < 0.5 {
                            continue;
                        }

                        for i in 0..HID_CHS {
                            let wi = row_idx * OUT_CHS + VIS_CHS + i;
                            out_buf[i] = f32::mul_add(neighbor_val, self.nca.weights[wi], out_buf[i]);
                        }
                    }
                }

                // Update hidden channels.
                for i in 0..HID_CHS {
                    next[(y, x, 2 * VIS_CHS + i)] = (next[(y, x, 2 * VIS_CHS + i)] + out_buf[i]).clamp(0.0, 1.0);
                }
            }
        }

        substrate.data = next;
    }

    // Updates RW channels
    pub fn update_rw(&mut self) {
        let substrate = &mut self.substrate;
        let mut next = substrate.data.clone();

        let w = substrate.width as i32;
        let h = substrate.height as i32;

        let data = substrate.data.view();
        let mut out_buf = [0.0; VIS_CHS];

        for y in 0..substrate.height {
            for x in 0..substrate.width {
                for i in 0..VIS_CHS {
                    out_buf[i] = self.nca.biases[i]
                }

                for (ni, (dx, dy)) in NHBD.iter().enumerate() {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || nx >= w || ny < 0 || ny >= h {
                        // Out of bounds
                        continue;
                    };

                    // Only rw and hidden
                    for ch_idx in VIS_CHS..INP_CHS {
                        let row_idx = ni * INP_CHS + ch_idx;
                        let neighbor_val = data[(ny as usize, nx as usize, ch_idx)];

                        // Alive masking
                        if neighbor_val < 0.5 {
                            continue;
                        }

                        for i in 0..VIS_CHS {
                            let wi = row_idx * OUT_CHS + i;
                            out_buf[i] = f32::mul_add(neighbor_val, self.nca.weights[wi], out_buf[i]);
                        }
                    }
                }

                // Update rw channels.
                for i in 0..VIS_CHS {
                    next[(y, x, VIS_CHS + i)] = (next[(y, x, VIS_CHS + i)] + out_buf[i]).clamp(0.0, 1.0);
                }
            }
        }

        substrate.data = next;
    }
}
