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
    pub sup_steps: usize,
    pub rec_steps: usize,
    pub hid_steps: usize,
    pub substrate: Substrate,
}

impl NCAExecutorCpu {
    pub fn new(nca: NCA, grid: &Grid) -> Self {
        let mut grid = grid.clone();

        nca.transform_pipeline.apply(&mut grid);
        let substrate = Substrate::from_grid(&grid);

        Self {
            nca,
            sup_steps: 0,
            rec_steps: 0,
            hid_steps: 0,
            substrate,
        }
    }

    pub fn run(&mut self) {
        loop {
            if self.step() {
                break;
            }
        }
    }

    /// Executes one iteration step.
    pub fn step(&mut self) -> bool {
        if self.sup_steps >= self.nca.sup_steps {
            return true;
        }

        if self.rec_steps >= self.nca.rec_steps {
            self.sup_steps += 1;
            self.rec_steps = 0;
            self.hid_steps = 0;
        } else if self.hid_steps >= self.nca.hid_steps {
            self.update_rw();
            self.rec_steps += 1;
            self.hid_steps = 0;
        } else {
            self.update_hidden();
            self.hid_steps += 1;
        }

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
                    next[(y, x, VIS_CHS + i)] = out_buf[i].clamp(0.0, 1.0);
                }
            }
        }

        substrate.data = next;
    }
}
