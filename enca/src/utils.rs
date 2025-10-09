use chrono::Local;
use std::{collections::HashSet, hash::Hash};

#[inline]
pub fn timestamp_for_dir() -> String {
    Local::now().format("%Y-%m-%d_%H-%M-%S").to_string()
}

pub fn union_sets<I, T>(it: I) -> HashSet<T>
where
    I: IntoIterator<Item = HashSet<T>>,
    T: Eq + Hash,
{
    let mut u: HashSet<T> = HashSet::new();
    for s in it {
        u.extend(s);
    }
    u
}

pub fn mean(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        panic!("Cannot compute mean of an empty array.")
    }

    xs.iter().sum::<f32>() / xs.len() as f32
}

pub fn stddev(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        panic!("Cannot compute standard deviation of an empty array.")
    }
    let m = mean(xs);
    let n = xs.len() as f32;
    let var = xs
        .iter()
        .map(|&x| {
            let d = x - m;
            d * d
        })
        .sum::<f32>()
        / n;
    var.sqrt()
}
