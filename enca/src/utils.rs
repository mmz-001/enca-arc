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
pub fn mean<T>(xs: &[T]) -> T
where
    T: Copy + std::iter::Sum<T> + std::ops::Div<Output = T> + From<f32>,
{
    if xs.is_empty() {
        panic!("Cannot compute mean of an empty array.")
    }

    xs.iter().copied().sum::<T>() / T::from(xs.len() as f32)
}
