use std::collections::{HashSet, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};
use std::ops::Index;

use serde::{Deserialize, Serialize};

/// ARC grid containing the 10 colors encoded as integers
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Grid {
    data: Vec<Vec<u8>>,
    width: usize,
    height: usize,
    colors: HashSet<u8>,
    hash: u64,
}

impl Grid {
    pub fn from_vec(data: Vec<Vec<u8>>) -> Self {
        let height = data.len();
        let width = data[0].len();
        let colors = HashSet::from_iter(data.clone().into_iter().flatten());

        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        let hash = hasher.finish();

        Self {
            data,
            width,
            height,
            colors,
            hash,
        }
    }

    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.height, self.width)
    }

    #[inline]
    pub fn get_hash(&self) -> u64 {
        self.hash
    }

    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    #[inline]
    pub fn colors(&self) -> &HashSet<u8> {
        &self.colors
    }

    #[inline]
    pub fn data(&self) -> &Vec<Vec<u8>> {
        &self.data
    }
}

impl Index<(usize, usize)> for Grid {
    type Output = u8;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.0][index.1]
    }
}
