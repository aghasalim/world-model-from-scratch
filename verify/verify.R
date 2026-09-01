# Exact inference on the claim the ablation is built on.
#
# The README argues from three seeds: at k=1 and k=5 the worst reconstruction
# seed still beats the best no-recon seed, so the bands do not overlap, and by
# k=10 they do. That is stated as an observation and never tested, and three
# seeds against three is small enough that the reader deserves to know what the
# strongest possible evidence at that size even is.
#
# With three against three there are choose(6, 3) = 20 ways to split the seeds,
# so the permutation distribution can be enumerated rather than sampled and the
# p value is exact. The floor on that p is computed below rather than assumed,
# because the difference of medians ties across several splits and the floor is
# not the 2/20 the count of splits suggests. Complete separation is the most
# three seeds can say, and it is worth saying out loud what that is worth.
#
# No packages, so CI needs nothing beyond base R.
#
# Run: Rscript verify/verify.R .

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) > 0) args[1] else "."

RECON <- "recon (Dreamer style)"
NORECON <- "no-recon (MuZero style)"
HORIZONS <- c(1, 5, 10, 15, 20, 40)
SEPARATED <- c(1, 5)          # the horizons the README claims do not overlap

ol <- read.csv(file.path(root, "results", "open-loop.csv"),
               check.names = FALSE, stringsAsFactors = FALSE)
stopifnot(all(c("mode", "seed", "step", "reward_mae") %in% names(ol)))

failures <- 0
splits <- combn(6, 3)

cat("recon against no-recon, exact permutation test over all",
    ncol(splits), "splits of 6 seeds\n")
cat(sprintf("  %-3s %8s %8s %9s %9s %8s  %s\n",
            "k", "recon", "norecon", "gap", "separated", "p", ""))

for (k in HORIZONS) {
    a <- ol$reward_mae[ol$mode == RECON & ol$step == k]
    b <- ol$reward_mae[ol$mode == NORECON & ol$step == k]
    if (length(a) != 3 || length(b) != 3) {
        cat(sprintf("  k=%d has %d and %d seeds, expected 3 and 3  FAIL\n",
                    k, length(a), length(b)))
        failures <- failures + 1
        next
    }
    pooled <- c(a, b)
    observed <- median(a) - median(b)
    stats <- apply(splits, 2, function(idx)
        median(pooled[idx]) - median(pooled[-idx]))
    p <- mean(abs(stats) >= abs(observed) - 1e-15)

    # Complete separation: every recon seed below every no-recon seed.
    separated <- max(a) < min(b)
    want <- k %in% SEPARATED
    ok <- separated == want
    failures <- failures + !ok
    cat(sprintf("  %-3d %8.4f %8.4f %+9.4f %9s %8.2f  %s\n",
                k, median(a), median(b), observed,
                if (separated) "yes" else "no", p,
                if (ok) "ok" else "FAIL, README says otherwise"))
}

# The other half of the README's argument: the advantage is gone by k=10, which
# is a statement about the spans overlapping, not about a p value.
span <- function(mode, k) range(ol$reward_mae[ol$mode == mode & ol$step == k])
lo_a <- span(RECON, 10); lo_b <- span(NORECON, 10)
overlap <- lo_a[2] >= lo_b[1] && lo_b[2] >= lo_a[1]
cat(sprintf("\nk=10 spans: recon %.4f to %.4f, no-recon %.4f to %.4f, overlap %s\n",
            lo_a[1], lo_a[2], lo_b[1], lo_b[2], if (overlap) "yes" else "no"))
if (!overlap) {
    cat("FAIL: the README says these overlap and they do not\n")
    failures <- failures + 1
}

# The floor on what three seeds can show, computed rather than asserted.
best_p <- min(apply(splits, 2, function(idx) {
    d <- median(seq_len(6)[idx]) - median(seq_len(6)[-idx])
    mean(abs(apply(splits, 2, function(j)
        median(seq_len(6)[j]) - median(seq_len(6)[-j]))) >= abs(d) - 1e-15)
}))
cat(sprintf("smallest two sided p reachable with 3 seeds against 3: %.2f\n", best_p))
if (best_p < 0.05) {
    cat("FAIL: the permutation enumeration is wrong, 3 against 3 cannot reach 0.05\n")
    failures <- failures + 1
}

if (failures > 0) {
    cat(sprintf("\n%d checks failed\n", failures))
    quit(status = 1)
}
cat("\nR confirms the separation at k=1 and k=5, the overlap from k=10 on,\n")
cat("and that complete separation is the most three seeds can show\n")
