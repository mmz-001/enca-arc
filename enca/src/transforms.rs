use std::fmt::Display;

use serde::{Deserialize, Serialize};

use crate::{constants::I_COL_MAP, grid::Grid};

#[derive(Serialize, Deserialize, Clone)]
pub struct Identity {}

#[derive(Serialize, Deserialize, Clone)]
pub struct Rotate90CW {}

impl Rotate90CW {
    pub fn apply(&self, grid: &mut Grid) {
        let width = grid.width();
        let height = grid.height();

        let mut rotated_data = vec![vec![0u8; height]; width];
        for y in 0..height {
            for x in 0..width {
                rotated_data[x][height - 1 - y] = grid[(y, x)];
            }
        }
        let rotated_grid = Grid::from_vec(rotated_data);

        *grid = rotated_grid;
    }

    pub fn revert(&self, grid: &mut Grid) {
        let width = grid.width();
        let height = grid.height();

        let mut rotated_data = vec![vec![0u8; height]; width];
        for y in 0..height {
            for x in 0..width {
                rotated_data[width - 1 - x][y] = grid[(y, x)];
            }
        }

        let rotated_grid = Grid::from_vec(rotated_data);

        *grid = rotated_grid;
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Rotate180 {}

impl Rotate180 {
    pub fn apply(&self, grid: &mut Grid) {
        let width = grid.width();
        let height = grid.height();
        let mut rotated = vec![vec![0u8; width]; height];
        for y in 0..height {
            for x in 0..width {
                rotated[height - 1 - y][width - 1 - x] = grid[(y, x)];
            }
        }
        *grid = Grid::from_vec(rotated);
    }

    pub fn revert(&self, grid: &mut Grid) {
        // 180-degree rotation is its own inverse
        self.apply(grid);
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Rotate270CW {}

impl Rotate270CW {
    pub fn apply(&self, grid: &mut Grid) {
        // 270 CW == 90 CCW => reuse Rotate90CW::revert
        Rotate90CW {}.revert(grid);
    }

    pub fn revert(&self, grid: &mut Grid) {
        // Inverse of 270 CW is 90 CW => reuse Rotate90CW::apply
        Rotate90CW {}.apply(grid);
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct FlipHorizontal {}

impl FlipHorizontal {
    pub fn apply(&self, grid: &mut Grid) {
        let width = grid.width();
        let height = grid.height();
        let mut flipped = vec![vec![0u8; width]; height];
        for y in 0..height {
            for x in 0..width {
                flipped[y][width - 1 - x] = grid[(y, x)];
            }
        }
        *grid = Grid::from_vec(flipped);
    }

    pub fn revert(&self, grid: &mut Grid) {
        // Horizontal flip is its own inverse
        self.apply(grid);
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct FlipVertical {}

impl FlipVertical {
    pub fn apply(&self, grid: &mut Grid) {
        let width = grid.width();
        let height = grid.height();
        let mut flipped = vec![vec![0u8; width]; height];
        for y in 0..height {
            for x in 0..width {
                flipped[height - 1 - y][x] = grid[(y, x)];
            }
        }
        *grid = Grid::from_vec(flipped);
    }

    pub fn revert(&self, grid: &mut Grid) {
        // Vertical flip is its own inverse
        self.apply(grid);
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ReflectMainDiagonal {}

impl ReflectMainDiagonal {
    pub fn apply(&self, grid: &mut Grid) {
        let width = grid.width();
        let height = grid.height();
        let mut reflected = vec![vec![0u8; height]; width];
        for y in 0..height {
            for x in 0..width {
                reflected[x][y] = grid[(y, x)];
            }
        }
        *grid = Grid::from_vec(reflected);
    }

    pub fn revert(&self, grid: &mut Grid) {
        // Reflection across main diagonal is its own inverse
        self.apply(grid);
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ReflectAntiDiagonal {}

impl ReflectAntiDiagonal {
    pub fn apply(&self, grid: &mut Grid) {
        let width = grid.width();
        let height = grid.height();
        let mut reflected = vec![vec![0u8; height]; width];
        for y in 0..height {
            for x in 0..width {
                reflected[width - 1 - x][height - 1 - y] = grid[(y, x)];
            }
        }
        *grid = Grid::from_vec(reflected);
    }

    pub fn revert(&self, grid: &mut Grid) {
        // Reflection across anti-diagonal is its own inverse
        self.apply(grid);
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct RemapColors {
    // Original -> mapped
    col_map: [u8; 10],
    // Mapped -> original
    rev_col_map: [u8; 10],
}

impl RemapColors {
    pub fn apply(&self, grid: &mut Grid) {
        self.remap_grid(grid, &self.col_map);
    }

    pub fn revert(&self, grid: &mut Grid) {
        self.remap_grid(grid, &self.rev_col_map);
    }
}

impl RemapColors {
    /// Creates new color map with identity mapping
    pub fn new() -> Self {
        RemapColors {
            col_map: I_COL_MAP,
            rev_col_map: I_COL_MAP,
        }
    }

    fn remap_grid(&self, grid: &mut Grid, map: &[u8; 10]) {
        let width = grid.width();
        let height = grid.height();

        let mut new_data = vec![vec![0u8; width]; height];

        for y in 0..height {
            for x in 0..width {
                new_data[y][x] = map[grid[(y, x)] as usize]
            }
        }

        *grid = Grid::from_vec(new_data);
    }

    pub fn map(&mut self, a: u8, b: u8) {
        self.col_map[a as usize] = b;
        self.rev_col_map[b as usize] = a;
    }
}

impl Default for RemapColors {
    fn default() -> Self {
        Self::new()
    }
}

impl Display for RemapColors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, col) in self.col_map.iter().enumerate() {
            writeln!(f, "{i} -> {col}")?
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub enum Transform {
    Identity(Identity),
    Rotate90CW(Rotate90CW),
    Rotate180(Rotate180),
    Rotate270CW(Rotate270CW),
    FlipHorizontal(FlipHorizontal),
    FlipVertical(FlipVertical),
    ReflectMainDiagonal(ReflectMainDiagonal),
    ReflectAntiDiagonal(ReflectAntiDiagonal),
    RemapColors(RemapColors),
}

impl Transform {
    pub fn apply(&self, grid: &mut Grid) {
        match self {
            Transform::Identity(_) => {}
            Transform::Rotate90CW(t) => t.apply(grid),
            Transform::Rotate180(t) => t.apply(grid),
            Transform::Rotate270CW(t) => t.apply(grid),
            Transform::FlipHorizontal(t) => t.apply(grid),
            Transform::FlipVertical(t) => t.apply(grid),
            Transform::ReflectMainDiagonal(t) => t.apply(grid),
            Transform::ReflectAntiDiagonal(t) => t.apply(grid),
            Transform::RemapColors(t) => t.apply(grid),
        }
    }

    pub fn revert(&self, grid: &mut Grid) {
        match self {
            Transform::Identity(_) => {}
            Transform::Rotate90CW(t) => t.revert(grid),
            Transform::Rotate180(t) => t.revert(grid),
            Transform::Rotate270CW(t) => t.revert(grid),
            Transform::FlipHorizontal(t) => t.revert(grid),
            Transform::FlipVertical(t) => t.revert(grid),
            Transform::ReflectMainDiagonal(t) => t.revert(grid),
            Transform::ReflectAntiDiagonal(t) => t.revert(grid),
            Transform::RemapColors(t) => t.revert(grid),
        }
    }
}

/// List of transforms that are applied sequentially applied
/// in order and sequentially reverted in reverse order.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct TransformPipeline {
    pub steps: Vec<Transform>,
}

impl TransformPipeline {
    pub fn apply(&self, grid: &mut Grid) {
        for transform in &self.steps {
            transform.apply(grid);
        }
    }

    pub fn revert(&self, grid: &mut Grid) {
        for transform in self.steps.iter().rev() {
            transform.revert(grid);
        }
    }
}
