use itertools::Itertools;

pub const ENCODING: [[f32; 4]; 10] = [
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
];

#[inline]
pub fn decode_color(encoded: &[f32]) -> u8 {
    let encoded = encoded.iter().map(|v| if *v > 0.5 { 1.0 } else { 0.0 }).collect_vec();

    let mut best_idx: u8 = 0;
    let mut best_dot = f32::NEG_INFINITY;
    for (i, proto) in ENCODING.iter().enumerate() {
        let dot = proto.iter().zip(encoded.iter()).map(|(a, b)| a * b).sum::<f32>();
        if dot > best_dot {
            best_dot = dot;
            best_idx = i as u8;
        }
    }
    best_idx
}
