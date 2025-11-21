pub use cmaes::objective_function::BatchObjectiveFunction;
pub use nalgebra::DVector;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct EvaluatedPoint {
    pub point: DVector<f64>,
    pub value: f64,
}

#[derive(Clone, Debug)]
pub enum TerminationReason {
    TargetFunctionValue,
    TolFunHist,
    MaxFunctionEvaluations,
    MinSigma,
    TimeLimit,
}

#[derive(Clone, Debug)]
pub struct TerminationData {
    pub overall_best: Option<EvaluatedPoint>,
    pub function_evals: usize,
    pub termination_reasons: Vec<TerminationReason>,
    pub best_values_history: Vec<f64>,
}

pub struct LMCMAOptions {
    x0: DVector<f64>,
    sigma0: f64,
    lambda: Option<usize>,
    fun_target: f64,
    max_function_evals: usize,
    tol_fun_hist: f64,
    min_sigma: f64,
    time_limit: Option<Duration>,
    seed: u64,
    verbose: bool,

    // LMCMA-specific options
    m: Option<usize>,
    base_m: usize,
    period: Option<usize>,
    n_steps: Option<usize>,
    c_c: Option<f64>,
    c_1: Option<f64>,
    c_s: f64,
    d_s: f64,
    z_star: f64,
}

impl LMCMAOptions {
    pub fn new(initial_mean: Vec<f64>, sigma0: f64) -> Self {
        Self {
            x0: DVector::from_vec(initial_mean),
            sigma0,
            lambda: None,
            fun_target: 1e-12,
            max_function_evals: 10_000,
            tol_fun_hist: 1e-12,
            min_sigma: 1e-12,
            time_limit: None,
            seed: 42,
            verbose: false,

            m: None,
            base_m: 4,
            period: None,
            n_steps: None,
            c_c: None,
            c_1: None,
            c_s: 0.3,
            d_s: 1.0,
            z_star: 0.3,
        }
    }

    pub fn tol_fun_hist(mut self, tol: f64) -> Self {
        self.tol_fun_hist = tol;
        self
    }

    pub fn fun_target(mut self, fun_target: f64) -> Self {
        self.fun_target = fun_target;
        self
    }

    pub fn lambda(mut self, lambda: Option<usize>) -> Self {
        self.lambda = lambda;
        self
    }

    pub fn max_function_evals(mut self, n: usize) -> Self {
        self.max_function_evals = n;
        self
    }

    pub fn min_sigma(mut self, s: f64) -> Self {
        self.min_sigma = s;
        self
    }

    pub fn time_limit(mut self, limit: Option<Duration>) -> Self {
        self.time_limit = limit;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    // LMCMA-specific builder methods (optional)
    pub fn m(mut self, m: Option<usize>) -> Self {
        self.m = m;
        self
    }
    pub fn base_m(mut self, base_m: usize) -> Self {
        self.base_m = base_m;
        self
    }
    pub fn period(mut self, period: Option<usize>) -> Self {
        self.period = period;
        self
    }
    pub fn n_steps(mut self, n_steps: Option<usize>) -> Self {
        self.n_steps = n_steps;
        self
    }
    pub fn c_c(mut self, c_c: Option<f64>) -> Self {
        self.c_c = c_c;
        self
    }
    pub fn c_1(mut self, c_1: Option<f64>) -> Self {
        self.c_1 = c_1;
        self
    }
    pub fn c_s(mut self, c_s: f64) -> Self {
        self.c_s = c_s;
        self
    }
    pub fn d_s(mut self, d_s: f64) -> Self {
        self.d_s = d_s;
        self
    }
    pub fn z_star(mut self, z_star: f64) -> Self {
        self.z_star = z_star;
        self
    }

    pub fn build<F: BatchObjectiveFunction>(self, f: F) -> Result<LMCMA<F>, String> {
        if self.x0.len() == 0 {
            return Err("LMCMAOptions: initial mean must be non-empty".to_string());
        }
        if !self.sigma0.is_finite() || self.sigma0 <= 0.0 {
            return Err("LMCMAOptions: initial sigma must be > 0".to_string());
        }
        Ok(LMCMA::new(self, f))
    }
}

pub struct LMCMA<F: BatchObjectiveFunction> {
    f: F,
    // options
    xmean: DVector<f64>,
    sigma: f64,
    fun_target: f64,
    max_function_evals: usize,
    tol_fun_hist: f64,
    min_sigma: f64,
    time_limit: Option<Duration>,
    verbose: bool,

    // rng
    rng: ChaCha8Rng,

    // CMA-ES recombination parameters
    lambda: usize,
    mu: usize,
    weights: Vec<f64>,

    // LMCMA parameters
    m: usize,
    base_m: usize,
    period: usize,
    n_steps: usize,
    c_s: f64,
    d_s: f64,
    z_star: f64,

    // precomputed constants
    a_const: f64, // sqrt(1 - c_1)
    c_const: f64, // 1/sqrt(1 - c_1)
    bd_2: f64,    // c_1/(1 - c_1)
    p_c_1: f64,   // 1 - c_c
    p_c_2: f64,   // sqrt(c_c*(2 - c_c)*mueff)

    // state
    p_c: DVector<f64>, // evolution path
    s_psr: f64,        // PSR accumulator

    vm: Vec<DVector<f64>>, // stored direction vectors (v)
    pm: Vec<DVector<f64>>, // stored p_c vectors (p)
    b: Vec<f64>,
    d: Vec<f64>,
    j: Vec<usize>, // order of slot indices
    l: Vec<usize>, // generation markers per slot
    it: usize,     // number of stored vectors

    rr: Vec<f64>, // PSR ranking weights length 2*lambda

    y_bak: Vec<f64>, // previous generation fitnesses

    fevals: usize,
    history_best: VecDeque<f64>,
    overall_best: Option<EvaluatedPoint>,
    start_time: Instant,
    n_generations: usize,
}

impl<F: BatchObjectiveFunction> LMCMA<F> {
    fn new(opts: LMCMAOptions, f: F) -> Self {
        let dim = opts.x0.len();

        let lambda = opts
            .lambda
            .unwrap_or((4.0 + (3.0 * (dim as f64).ln()).floor()) as usize);
        let lambda = lambda.max(2);
        let mu = (lambda / 2).max(1);

        // recombination weights
        let mut weights = vec![0.0; mu];
        for i in 0..mu {
            weights[i] = (mu as f64 + 0.5).ln() - ((i + 1) as f64).ln();
        }
        let wsum: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= wsum;
        }
        let mueff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();

        // LMCMA specific defaults
        let m = opts.m.unwrap_or_else(|| (4.0 + 3.0 * (dim as f64).ln()) as usize);
        let base_m = opts.base_m;
        let period = opts.period.unwrap_or_else(|| (dim as f64).ln().max(1.0) as usize);
        let n_steps = opts.n_steps.unwrap_or(dim);
        let c_c = opts.c_c.unwrap_or(0.5 / (dim as f64).sqrt());
        let c_1 = opts.c_1.unwrap_or(1.0 / (10.0 * (dim as f64 + 1.0).ln()));
        let c_s = opts.c_s;
        let d_s = opts.d_s;
        let z_star = opts.z_star;

        let a_const = (1.0 - c_1).sqrt();
        let c_const = 1.0 / (1.0 - c_1).sqrt();
        let bd_2 = c_1 / (1.0 - c_1);
        let p_c_1 = 1.0 - c_c;
        let p_c_2 = (c_c * (2.0 - c_c) * mueff).sqrt();

        let rng = ChaCha8Rng::seed_from_u64(opts.seed);

        let mut vm = Vec::with_capacity(m);
        let mut pm = Vec::with_capacity(m);
        for _ in 0..m {
            vm.push(DVector::zeros(dim));
            pm.push(DVector::zeros(dim));
        }
        let b = vec![0.0; m];
        let d = vec![0.0; m];

        let j = (0..m).collect::<Vec<_>>(); // order slots; validity controlled by it
        let l = vec![0usize; m];
        let it = 0usize;

        let rr: Vec<f64> = (0..(2 * lambda)).rev().map(|i| i as f64).collect();

        let x0 = opts.x0.clone();
        let mut es = Self {
            f,
            xmean: x0,
            sigma: opts.sigma0,
            fun_target: opts.fun_target,
            max_function_evals: opts.max_function_evals,
            tol_fun_hist: opts.tol_fun_hist,
            min_sigma: opts.min_sigma,
            time_limit: opts.time_limit,
            verbose: opts.verbose,

            rng,

            lambda,
            mu,
            weights,

            m,
            base_m,
            period,
            n_steps,
            c_s,
            d_s,
            z_star,

            a_const,
            c_const,
            bd_2,
            p_c_1,
            p_c_2,

            p_c: DVector::zeros(dim),
            s_psr: 0.0,

            vm,
            pm,
            b,
            d,
            j,
            l,
            it,

            rr,

            y_bak: vec![f64::INFINITY; lambda],

            fevals: 0,
            history_best: VecDeque::new(),
            overall_best: None,
            start_time: Instant::now(),
            n_generations: 0,
        };

        es.y_bak = vec![f64::INFINITY; es.lambda];

        es
    }

    fn update_overall_best(&mut self, x: DVector<f64>, fx: f64) {
        if let Some(best) = &self.overall_best {
            if fx < best.value {
                self.overall_best = Some(EvaluatedPoint { point: x, value: fx });
            }
        } else {
            self.overall_best = Some(EvaluatedPoint { point: x, value: fx });
        }
    }

    fn push_best_history(&mut self, v: f64) {
        self.history_best.push_front(v);
    }

    fn past_generations_a(&self, dim: usize) -> usize {
        let lambda = self.lambda as f64;
        10 + (30.0 * dim as f64 / lambda).ceil() as usize
    }

    fn check_tol_fun_hist(&self, dim: usize) -> bool {
        let need = self.past_generations_a(dim);
        if self.history_best.len() >= need {
            if let Some(r) = range(self.history_best.iter().take(need).cloned()) {
                return r < self.tol_fun_hist;
            }
        }
        false
    }

    fn rademacher(&mut self, dim: usize) -> DVector<f64> {
        let mut v = DVector::zeros(dim);
        for i in 0..dim {
            v[i] = if self.rng.random_bool(0.5) { 1.0 } else { -1.0 };
        }
        v
    }

    // Az: Cholesky factor-vector update applied to z
    fn az(&self, z: &DVector<f64>, start: usize, it: usize) -> DVector<f64> {
        let dim = z.len();
        let mut x = z.clone();
        for t in start..it {
            let slot = self.j[t];
            // dot(vm[slot], z)
            let dot = self.vm[slot].dot(z);
            // x = a*x + b[slot] * dot * pm[slot]
            let mut add = self.pm[slot].clone();
            add *= self.b[slot] * dot;
            x *= self.a_const;
            x += add;
        }
        // guard against NaN/Inf
        for i in 0..dim {
            if !x[i].is_finite() {
                x[i] = 0.0;
            }
        }
        x
    }

    // Ainvz: inverse Cholesky factor-vector update
    fn a_inv_z(&self, v: &DVector<f64>, i: usize) -> DVector<f64> {
        let dim = v.len();
        let mut x = v.clone();
        for t in 0..i {
            let slot = self.j[t];
            // x = c*x - d[slot] * dot(vm[slot], x) * vm[slot]
            let dot = self.vm[slot].dot(&x);
            let mut sub = self.vm[slot].clone();
            sub *= self.d[slot] * dot;
            x *= self.c_const;
            x -= sub;
        }
        for i in 0..dim {
            if !x[i].is_finite() {
                x[i] = 0.0;
            }
        }
        x
    }

    pub fn run_batch(&mut self) -> TerminationData {
        let dim = self.xmean.len();

        let mut reasons = Vec::new();

        'outer: loop {
            // Check global terminations
            if let Some(limit) = self.time_limit {
                if self.start_time.elapsed() >= limit {
                    reasons.push(TerminationReason::TimeLimit);
                    break 'outer;
                }
            }
            if self.sigma <= self.min_sigma {
                reasons.push(TerminationReason::MinSigma);
                break 'outer;
            }
            if self.fevals >= self.max_function_evals {
                reasons.push(TerminationReason::MaxFunctionEvaluations);
                break 'outer;
            }
            if self.check_tol_fun_hist(dim) {
                reasons.push(TerminationReason::TolFunHist);
                break 'outer;
            }

            // Sampling and evaluation with mirrored sampling
            let mut ar_x: Vec<DVector<f64>> = Vec::with_capacity(self.lambda);

            let mut sign: f64 = 1.0;
            let mut a_z: DVector<f64> = DVector::zeros(dim);

            for k in 0..self.lambda {
                if sign > 0.0 {
                    // direction vectors selection
                    let base_scale = if k == 0 {
                        10.0 * self.base_m as f64
                    } else {
                        self.base_m as f64
                    };
                    let gauss: f64 = StandardNormal.sample(&mut self.rng);
                    let mut base_m = base_scale * gauss.abs();
                    if base_m > self.it as f64 {
                        base_m = self.it as f64;
                    }
                    let start = if self.it > 1 {
                        let s = (self.it as f64 - base_m).floor() as isize;
                        if s > 0 { s as usize } else { 0usize }
                    } else {
                        0usize
                    };

                    // z ~ Rademacher
                    let z = self.rademacher(dim);
                    a_z = self.az(&z, start, self.it);
                }

                // x_k = mean + sign * sigma * a_z
                let mut x = self.xmean.clone();
                for i in 0..dim {
                    x[i] += sign * self.sigma * a_z[i];
                }
                ar_x.push(x);

                // flip sign for mirrored sampling
                sign = -sign;
            }

            let ar_fitness = self.f.evaluate_batch(&ar_x);
            self.fevals += ar_x.len();

            // Sorting by fitness
            let mut idx: Vec<usize> = (0..self.lambda).collect();
            idx.sort_by(|&a, &b| ar_fitness[a].partial_cmp(&ar_fitness[b]).unwrap());

            let gen_best_val = ar_fitness[idx[0]];
            let gen_best_x = ar_x[idx[0]].clone();
            self.push_best_history(gen_best_val);
            self.update_overall_best(gen_best_x.clone(), gen_best_val);

            if gen_best_val <= self.fun_target {
                reasons.push(TerminationReason::TargetFunctionValue);
                break 'outer;
            }

            // Weighted recombination to update mean
            let mut mean_bak = DVector::zeros(dim);
            for i in 0..self.mu {
                let ii = idx[i];
                let w = self.weights[i];
                for d in 0..dim {
                    mean_bak[d] += w * ar_x[ii][d];
                }
            }

            // Evolution path update
            // p_c = (1 - c_c)*p_c + sqrt(c_c*(2-c_c)*mueff) * (mean_bak - xmean)/sigma
            let mut diff = &mean_bak - &self.xmean;
            diff /= self.sigma;
            self.p_c *= self.p_c_1;
            self.p_c.axpy(self.p_c_2, &diff, 1.0);

            // Direction vectors selection and storage
            if self.n_generations % self.period == 0 {
                let ng = self.n_generations / self.period;
                let mut i_min = 1usize;

                if ng < self.m {
                    // store in increasing order
                    self.j[ng] = ng;
                } else if self.m > 1 {
                    // choose pair with distance closest to n_steps
                    let mut d_min = (self.l[self.j[1]] as isize - self.l[self.j[0]] as isize) - self.n_steps as isize;
                    for j in 2..self.m {
                        let d_cur =
                            (self.l[self.j[j]] as isize - self.l[self.j[j - 1]] as isize) - self.n_steps as isize;
                        if d_cur < d_min {
                            d_min = d_cur;
                            i_min = j;
                        }
                    }
                    i_min = if d_min >= 0 { 0 } else { i_min };
                    // rotate order
                    let updated = self.j[i_min];
                    for j in i_min..(self.m - 1) {
                        self.j[j] = self.j[j + 1];
                    }
                    self.j[self.m - 1] = updated;
                }

                self.it = usize::min(self.m, ng + 1);
                // record generation marker
                let last_slot = self.j[self.it - 1];
                self.l[last_slot] = ng * self.period;

                // store current p_c into pm[last_slot]
                self.pm[last_slot] = self.p_c.clone();

                // compute vm, and update b, d for involved indices
                let start_i = if i_min == 1 { 0 } else { i_min };
                for i in start_i..self.it {
                    let slot = self.j[i];
                    let v = self.a_inv_z(&self.pm[slot], i);
                    self.vm[slot] = v;

                    let v_n = self.vm[slot].dot(&self.vm[slot]).max(1e-32);
                    let bd_3 = (1.0 + self.bd_2 * v_n).sqrt();

                    self.b[slot] = self.a_const / v_n * (bd_3 - 1.0);
                    self.d[slot] = self.c_const / v_n * (1.0 - 1.0 / bd_3);
                }
            }

            // PSR step-size adaptation
            if self.n_generations > 0 {
                // concatenate current and previous y
                let mut merged: Vec<(usize, f64, bool)> = Vec::with_capacity(2 * self.lambda);
                for (i, &y) in ar_fitness.iter().enumerate() {
                    merged.push((i, y, true)); // true => current
                }
                for (i, &y) in self.y_bak.iter().enumerate() {
                    merged.push((self.lambda + i, y, false)); // false => previous
                }
                // sort by fitness
                merged.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                // rr is precomputed: length 2*lambda; rr[j] aligned to rank position j
                let mut sum_curr = 0.0;
                let mut sum_prev = 0.0;
                for (rank_pos, (_, _, is_curr)) in merged.iter().enumerate() {
                    let rr_val = self.rr[rank_pos];
                    if *is_curr {
                        sum_curr += rr_val;
                    } else {
                        sum_prev += rr_val;
                    }
                }
                let mut z_psr = (sum_curr - sum_prev) / (self.lambda as f64).powi(2);
                z_psr -= self.z_star;

                self.s_psr = (1.0 - self.c_s) * self.s_psr + self.c_s * z_psr;
                self.sigma *= (self.s_psr / self.d_s).exp();
            }

            // set previous fitness to current
            self.y_bak = ar_fitness.clone();

            // finalize mean update
            self.xmean = mean_bak;

            // Verbose logging per generation
            if self.verbose && self.fevals % (self.lambda * 50) == 0 {
                if let Some(best) = &self.overall_best {
                    println!(
                        "LM-CMA: fevals={} best={:.3e} sigma={:.3e}",
                        self.fevals, best.value, self.sigma
                    );
                }
            }

            self.n_generations += 1;
        }

        TerminationData {
            overall_best: self.overall_best.clone(),
            function_evals: self.fevals,
            termination_reasons: reasons,
            best_values_history: self.history_best.iter().cloned().collect(),
        }
    }
}

fn range<I: Iterator<Item = f64>>(mut it: I) -> Option<f64> {
    let mut minv = it.next()?;
    let mut maxv = minv;
    for v in it {
        if v < minv {
            minv = v;
        } else if v > maxv {
            maxv = v;
        }
    }
    Some(maxv - minv)
}
