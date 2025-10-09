use rand::Rng;

pub enum Optimize {
    Maximize,
    Minimize,
}

/// Tournament selection with replacement
pub struct TournamentSelector {
    k: usize,
    mode: Optimize,
}

impl TournamentSelector {
    pub fn new(k: usize, mode: Optimize) -> Self {
        assert!(k >= 1, "Tournament size k must be >= 1");

        Self { k, mode }
    }

    pub fn select<I: Score + Clone>(&self, population: &[I], rng: &mut impl Rng) -> Vec<I> {
        assert!(!population.is_empty(), "Population must not be empty");

        let n = population.len();
        let mut selected = Vec::with_capacity(n);

        for _ in 0..n {
            // Initialize with a first random pick
            let mut best_idx = rng.random_range(0..n);
            let mut best_score = population[best_idx].score();

            // Sample remaining k-1 contestants
            for _ in 1..self.k {
                let idx = rng.random_range(0..n);
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

            selected.push(population[best_idx].clone());
        }

        selected
    }
}

pub trait Score {
    fn score(&self) -> f32;
}
