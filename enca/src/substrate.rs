use crate::{
    color::{EMBEDDING, decode_color},
    constants::{INP_CHS, RO_CH_RNG, RW_CH_RNG},
    grid::Grid,
};
use ndarray::{Array3, s};

/// The lattice with all visible and hidden channels that the NCA operates on.
#[derive(Clone, Debug)]
pub struct Substrate {
    pub data: Array3<f32>,
    pub width: usize,
    pub height: usize,
}

impl Substrate {
    pub fn from_grid(grid: &Grid) -> Self {
        let height = grid.height();
        let width = grid.width();
        let mut data = Array3::<f32>::zeros((height, width, INP_CHS));
        let embedding = &*EMBEDDING;

        for yi in 0..height {
            for xi in 0..width {
                let v = grid[(yi, xi)];
                for i in RO_CH_RNG {
                    data[(yi, xi, i)] = embedding[(v as usize, i)]
                }
            }
        }

        Self { data, width, height }
    }

    pub fn to_grid(&self) -> Grid {
        let mut grid_data = vec![vec![0u8; self.width]; self.height];

        for yi in 0..self.height {
            for xi in 0..self.width {
                // Only RW visible channels are used
                let v = self.data.slice(s![yi, xi, RW_CH_RNG]);
                grid_data[yi][xi] = decode_color(v.as_slice().unwrap());
            }
        }

        Grid::from_vec(grid_data)
    }
}
