use rand::{Rng, seq::SliceRandom};

pub enum Optimize {
    Maximize,
    Minimize,
}

/// Tournament selection without replacement (global partition into tournaments).
pub struct TournamentSelector {
    k: usize,
    mode: Optimize,
}

impl TournamentSelector {
    pub fn new(k: usize, mode: Optimize) -> Self {
        assert!(k >= 1, "Tournament size k must be >= 1");
        Self { k, mode }
    }

    pub fn select<'a, I: Score>(&self, population: &'a [&'a I], rng: &mut impl Rng) -> Vec<&'a I> {
        if population.is_empty() {
            return vec![];
        }

        let n = population.len();
        let mut indices: Vec<usize> = (0..n).collect();

        indices.shuffle(rng);

        let mut selected = Vec::with_capacity(n.div_ceil(self.k));

        for chunk in indices.chunks(self.k) {
            let mut best_idx = chunk[0];
            let mut best_score = population[best_idx].score();

            for &idx in &chunk[1..] {
                let score = population[idx].score();
                let better = match self.mode {
                    Optimize::Maximize => score > best_score,
                    Optimize::Minimize => score < best_score,
                };
                if better {
                    best_idx = idx;
                    best_score = score;
                }
            }

            selected.push(population[best_idx]);
        }

        selected
    }
}
pub trait Score {
    fn score(&self) -> f32;
}
