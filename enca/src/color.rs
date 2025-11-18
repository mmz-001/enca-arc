use ndarray::{Array2, arr1, arr2};
use ndarray_stats::QuantileExt;
use std::sync::LazyLock;

pub static EMBEDDING: LazyLock<Array2<f32>> = LazyLock::new(|| {
    arr2(&[
        [0., 0., 0., 0.],
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
        [1., 0., 1., 0.],
        [1., 0., 0., 1.],
        [0., 1., 1., 0.],
        [0., 1., 0., 1.],
        [0., 0., 1., 1.],
    ])
});

#[inline]
pub fn decode_color(encoded: &[f32]) -> u8 {
    let encoded = arr1(encoded).mapv_into(|v| if v < 0.5 { 0.0 } else { v });
    let embedding = &*EMBEDDING;
    embedding.dot(&encoded).argmax().unwrap() as u8
}
