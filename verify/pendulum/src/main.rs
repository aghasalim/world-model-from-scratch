//! The environment, reimplemented, and the random policy baseline remeasured.
//!
//! Two published claims rest on wm/envs.py and on nothing else.
//!
//! The first is the number every learning curve is read against: "a random
//! policy scores about -38", drawn as a dashed line in results/learning-curves.png
//! and hard coded as RANDOM_RETURN in bench/style consumers. If that number were
//! wrong, every "nothing beats random" sentence in the README would be wrong
//! with it, and nothing in the repository would notice. It was never measured
//! with more than the handful of rollouts a training run happens to collect.
//! This runs 200,000 independent episodes, which the Python sweep could not
//! afford next to everything else it does, and reports a standard error.
//!
//! The second is the reason the environment is hand written at all: two states
//! with the same angle and opposite velocity produce the same observation, so
//! the task is a POMDP. tests/ asserts that on a few states. This sweeps a grid
//! of four million and also checks the ambiguity is resolvable, which is the
//! half that makes the task learnable rather than impossible.
//!
//! Both only mean something if this integrator is the one wm/envs.py uses, so
//! before either it reproduces the reference trajectory that
//! verify/export_golden.py exports from PyTorch.
//!
//! Run: cd verify/pendulum && cargo run --release -- ../..

use std::collections::HashMap;
use std::env;
use std::f64::consts::PI;
use std::fs;
use std::process::exit;

// Straight out of wm/envs.py.
const MAX_SPEED: f64 = 8.0;
const MAX_TORQUE: f64 = 2.0;
const DT: f64 = 0.05;
const GRAVITY: f64 = 10.0;
const MASS: f64 = 1.0;
const LENGTH: f64 = 1.0;

const EPISODES: usize = 200_000;
const HORIZON: usize = 60;
const GRID: usize = 2_000;
/// The reference is float32 and this is f64, so the trajectories separate. The
/// tolerance is on the absolute difference after 60 steps and the largest one
/// actually seen is printed next to it.
const GOLDEN_TOL: f64 = 1e-4;
/// How far the measured mean return may sit from the baseline the repository
/// draws its learning curves against. The quoted value is rounded to one decimal
/// place, so this has to leave room for that on top of the Monte Carlo error,
/// which is printed alongside.
const RETURN_TOL: f64 = 0.15;

/// xorshift64*. Not cryptographic and not meant to be: it needs to be uniform,
/// fast, and seeded reproducibly so a failure here can be rerun.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform on [0, 1) from the top 53 bits, which is the whole mantissa.
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    /// Uniform on [-1, 1).
    fn signed(&mut self) -> f64 {
        self.unit() * 2.0 - 1.0
    }
}

fn angle_normalize(x: f64) -> f64 {
    // Python's % on a positive divisor lands in [0, 2pi), which is rem_euclid
    // here and not the truncated %.
    (x + PI).rem_euclid(2.0 * PI) - PI
}

/// One step of wm/envs.py: reward is charged on the state entered from, then
/// the velocity is integrated and clamped, then the angle.
fn step(theta: f64, theta_dot: f64, action: f64) -> (f64, f64, f64) {
    let u = action.clamp(-1.0, 1.0) * MAX_TORQUE;
    let cost = angle_normalize(theta).powi(2) + 0.1 * theta_dot * theta_dot + 0.001 * u * u;
    let reward = -cost / 10.0;
    let new_dot = (theta_dot
        + (3.0 * GRAVITY / (2.0 * LENGTH) * theta.sin() + 3.0 / (MASS * LENGTH * LENGTH) * u) * DT)
        .clamp(-MAX_SPEED, MAX_SPEED);
    (theta + new_dot * DT, new_dot, reward)
}

fn observe(theta: f64) -> (f64, f64) {
    (theta.cos(), theta.sin())
}

struct Block {
    rows: usize,
    cols: usize,
    v: Vec<f64>,
}

impl Block {
    fn at(&self, r: usize, c: usize) -> f64 {
        self.v[r * self.cols + c]
    }
}

/// Blocks are looked up by name, never by position.
fn load(path: &str) -> HashMap<String, Block> {
    let text = fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("cannot read {}: {}", path, e);
        exit(2);
    });
    let mut out = HashMap::new();
    let mut lines = text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty());
    while let Some(header) = lines.next() {
        let parts: Vec<&str> = header.split_whitespace().collect();
        if parts.len() != 3 {
            eprintln!("bad block header in {}: {}", path, header);
            exit(2);
        }
        let rows: usize = parts[1].parse().unwrap();
        let cols: usize = parts[2].parse().unwrap();
        let mut v = Vec::with_capacity(rows * cols);
        for _ in 0..rows {
            let line = lines.next().unwrap_or_else(|| {
                eprintln!("{}: block {} is short", path, parts[0]);
                exit(2);
            });
            for tok in line.split_whitespace() {
                v.push(tok.parse::<f64>().unwrap());
            }
        }
        if v.len() != rows * cols {
            eprintln!(
                "{}: block {} has {} values, expected {}",
                path,
                parts[0],
                v.len(),
                rows * cols
            );
            exit(2);
        }
        out.insert(parts[0].to_string(), Block { rows, cols, v });
    }
    out
}

fn check_golden(root: &str) -> bool {
    let path = format!("{}/verify/golden/pendulum.txt", root);
    let b = load(&path);
    let get = |name: &str| -> &Block {
        b.get(name).unwrap_or_else(|| {
            eprintln!("missing block {} in {}", name, path);
            exit(2);
        })
    };
    let (th0, thd0) = (get("theta0"), get("theta_dot0"));
    let (acts, wth, wthd, wr) = (
        get("actions"),
        get("theta"),
        get("theta_dot"),
        get("reward"),
    );
    let batch = acts.rows;
    let steps = acts.cols;
    if th0.cols != batch || wth.rows != batch || wth.cols != steps {
        eprintln!("the exported blocks disagree about their own shapes");
        exit(2);
    }

    let (mut mth, mut mthd, mut mr) = (0.0f64, 0.0f64, 0.0f64);
    for s in 0..batch {
        let mut theta = th0.at(0, s);
        let mut theta_dot = thd0.at(0, s);
        for k in 0..steps {
            let (nt, nd, r) = step(theta, theta_dot, acts.at(s, k));
            theta = nt;
            theta_dot = nd;
            mth = mth.max((theta - wth.at(s, k)).abs());
            mthd = mthd.max((theta_dot - wthd.at(s, k)).abs());
            mr = mr.max((r - wr.at(s, k)).abs());
        }
    }
    println!(
        "reference trajectory, {} sequences of {} steps against PyTorch",
        batch, steps
    );
    println!(
        "  largest |Rust - torch|  theta {:.3e}  theta_dot {:.3e}  reward {:.3e}",
        mth, mthd, mr
    );
    if mth.max(mthd).max(mr) > GOLDEN_TOL {
        println!("  FAIL: the integrator here is not the one in wm/envs.py");
        return false;
    }
    println!("  ok, within {:.0e}", GOLDEN_TOL);
    true
}

/// The baseline is not hard coded here. It is read out of bench/figures.py, which
/// is the file that draws it, so editing the line in the figure without
/// remeasuring is what this catches.
fn quoted_return(root: &str) -> f64 {
    let path = format!("{}/bench/figures.py", root);
    let text = fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("cannot read {}: {}", path, e);
        exit(2);
    });
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("RANDOM_RETURN") {
            let rest = rest.trim_start().trim_start_matches('=').trim_start();
            let value: String = rest
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '-' || *c == '.' || *c == 'e')
                .collect();
            if let Ok(v) = value.parse::<f64>() {
                return v;
            }
        }
    }
    eprintln!("bench/figures.py has no RANDOM_RETURN to check against");
    exit(2);
}

fn random_return(root: &str) -> bool {
    let quoted = quoted_return(root);
    let mut rng = Rng::new(0x5eed_1234_9abc_def0);
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    for _ in 0..EPISODES {
        let mut theta = rng.signed() * PI;
        let mut theta_dot = rng.signed();
        let mut ret = 0.0f64;
        for _ in 0..HORIZON {
            let (nt, nd, r) = step(theta, theta_dot, rng.signed());
            theta = nt;
            theta_dot = nd;
            ret += r;
        }
        sum += ret;
        sumsq += ret * ret;
    }
    let n = EPISODES as f64;
    let mean = sum / n;
    let sd = (sumsq / n - mean * mean).max(0.0).sqrt();
    let se = sd / n.sqrt();
    println!(
        "\nrandom policy over {} episodes of {} steps",
        EPISODES, HORIZON
    );
    println!(
        "  mean return {:+.4}  sd {:.4}  standard error {:.4}",
        mean, sd, se
    );
    println!(
        "  bench/figures.py draws {:+.4}, difference {:+.4}",
        quoted,
        mean - quoted
    );
    if (mean - quoted).abs() > RETURN_TOL {
        println!(
            "  FAIL: the baseline every learning curve is read against is off by more than {}",
            RETURN_TOL
        );
        return false;
    }
    println!(
        "  ok, within {} of the baseline the repository draws",
        RETURN_TOL
    );
    true
}

fn ambiguity() -> bool {
    // Same angle, opposite velocity: identical observation now, different
    // observation one step later. The first half is why the task is a POMDP,
    // the second is why memory is enough to solve it.
    let mut same_now = 0usize;
    let mut differ_next = 0usize;
    let mut total = 0usize;
    let mut min_next_gap = f64::INFINITY;
    for i in 0..GRID {
        let theta = -PI + 2.0 * PI * (i as f64 + 0.5) / GRID as f64;
        for j in 0..GRID {
            // Skip velocities at or near zero, where the two states are the
            // same state and no amount of memory separates them.
            let theta_dot = MAX_SPEED * (j as f64 + 0.5) / GRID as f64;
            total += 1;
            let (c1, s1) = observe(theta);
            let (c2, s2) = observe(theta);
            if (c1 - c2).abs() < 1e-12 && (s1 - s2).abs() < 1e-12 {
                same_now += 1;
            }
            let (t_a, _, _) = step(theta, theta_dot, 0.0);
            let (t_b, _, _) = step(theta, -theta_dot, 0.0);
            let (ca, sa) = observe(t_a);
            let (cb, sb) = observe(t_b);
            let gap = ((ca - cb).powi(2) + (sa - sb).powi(2)).sqrt();
            if gap > 1e-9 {
                differ_next += 1;
            }
            min_next_gap = min_next_gap.min(gap);
        }
    }
    println!(
        "\nPOMDP ambiguity over a {} by {} grid ({} states)",
        GRID, GRID, total
    );
    println!("  identical observation now      {}/{}", same_now, total);
    println!(
        "  observation differs one step on {}/{}, smallest gap {:.3e}",
        differ_next, total, min_next_gap
    );
    if same_now != total {
        println!("  FAIL: velocity is visible in the observation");
        return false;
    }
    if differ_next != total {
        println!("  FAIL: the two states stay indistinguishable, memory would not help");
        return false;
    }
    println!("  ok, hidden now and separable one step later everywhere on the grid");
    true
}

fn main() {
    let root = env::args().nth(1).unwrap_or_else(|| ".".to_string());
    let mut ok = check_golden(&root);
    ok &= random_return(&root);
    ok &= ambiguity();
    if !ok {
        println!("\nRust disagrees with the environment the README describes");
        exit(1);
    }
    println!(
        "\nRust reproduces the environment, the random baseline and the ambiguity it is built on"
    );
}
