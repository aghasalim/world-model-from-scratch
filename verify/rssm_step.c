/* Recompute the RSSM dynamics step in C and require it to match PyTorch.
 *
 * Every open loop number in the README comes out of one loop, run with no
 * observations to correct it:
 *
 *     h = GRUCell([z, a], h)
 *     z = mean of prior_net(h)
 *     r = reward_head([h, z])
 *
 * That loop exists once, in wm/rssm.py and experiments/main.py, so nothing has
 * ever disagreed with it. This is a second implementation written from the
 * definitions of a GRU cell, an ELU and a softplus rather than from the torch
 * source, reading the weights and the reference rollout that
 * verify/export_golden.py writes.
 *
 * The reference is float32 and this is double, so exact agreement is not
 * available and is not asked for. The tolerance below is on the absolute
 * difference and is reported alongside the largest one actually seen.
 *
 *     cc -std=c99 -O2 -o rssm_step verify/rssm_step.c -lm && ./rssm_step .
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOL 2e-5
#define MAX_BLOCKS 32
#define NAME_LEN 64

typedef struct {
    char name[NAME_LEN];
    int rows, cols;
    double *v;
} Block;

typedef struct {
    Block b[MAX_BLOCKS];
    int n;
} Blocks;

static double at(const Block *b, int r, int c) { return b->v[(size_t)r * b->cols + c]; }

static const Block *get(const Blocks *bs, const char *name)
{
    for (int i = 0; i < bs->n; i++)
        if (strcmp(bs->b[i].name, name) == 0)
            return &bs->b[i];
    fprintf(stderr, "missing block %s\n", name);
    exit(2);
}

/* Blocks are looked up by name, never by position, so a block added to the
 * export cannot silently shift what this reads. */
static void load(Blocks *bs, const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    bs->n = 0;
    char name[NAME_LEN];
    int rows, cols, ch;
    for (;;) {
        ch = fgetc(f);
        if (ch == EOF) break;
        if (ch == '#') {
            while ((ch = fgetc(f)) != EOF && ch != '\n') {
                /* skip the comment line */
            }
            continue;
        }
        if (ch == '\n' || ch == ' ') continue;
        ungetc(ch, f);
        if (fscanf(f, "%63s %d %d", name, &rows, &cols) != 3) break;
        if (bs->n >= MAX_BLOCKS) { fprintf(stderr, "too many blocks\n"); exit(2); }
        Block *b = &bs->b[bs->n++];
        strncpy(b->name, name, NAME_LEN - 1);
        b->name[NAME_LEN - 1] = '\0';
        b->rows = rows; b->cols = cols;
        b->v = malloc(sizeof(double) * (size_t)rows * cols);
        if (!b->v) { fprintf(stderr, "out of memory\n"); exit(2); }
        for (size_t i = 0; i < (size_t)rows * cols; i++)
            if (fscanf(f, "%lf", &b->v[i]) != 1) {
                fprintf(stderr, "%s: short block %s\n", path, name);
                exit(2);
            }
    }
    fclose(f);
}

static double sigmoidd(double x) { return 1.0 / (1.0 + exp(-x)); }
static double elu(double x) { return x > 0.0 ? x : expm1(x); }
/* torch.nn.functional.softplus is linear above beta*x = 20, and saying so here
 * rather than relying on log1p not overflowing keeps the two the same function. */
static double softplus(double x) { return x > 20.0 ? x : log1p(exp(x)); }

/* out = W x + b, W is (out_dim, in_dim) as torch stores it. */
static void affine(const Block *W, const Block *bias, const double *x, double *out)
{
    for (int i = 0; i < W->rows; i++) {
        double s = bias ? at(bias, 0, i) : 0.0;
        for (int j = 0; j < W->cols; j++)
            s += at(W, i, j) * x[j];
        out[i] = s;
    }
}

int main(int argc, char **argv)
{
    const char *root = argc > 1 ? argv[1] : ".";
    char path[1024];
    Blocks W, G;
    W.n = 0;
    G.n = 0;

    snprintf(path, sizeof path, "%s/verify/golden/rssm_weights.txt", root);
    load(&W, path);
    snprintf(path, sizeof path, "%s/verify/golden/rssm_rollout.txt", root);
    load(&G, path);

    const Block *w_ih = get(&W, "cell.weight_ih");
    const Block *w_hh = get(&W, "cell.weight_hh");
    const Block *b_ih = get(&W, "cell.bias_ih");
    const Block *b_hh = get(&W, "cell.bias_hh");
    const Block *p0w = get(&W, "prior.0.weight"), *p0b = get(&W, "prior.0.bias");
    const Block *p2w = get(&W, "prior.2.weight"), *p2b = get(&W, "prior.2.bias");
    const Block *r0w = get(&W, "reward.0.weight"), *r0b = get(&W, "reward.0.bias");
    const Block *r2w = get(&W, "reward.2.weight"), *r2b = get(&W, "reward.2.bias");

    const Block *h0 = get(&G, "h0"), *z0 = get(&G, "z0");
    const Block *acts = get(&G, "actions");
    const Block *want_h = get(&G, "h"), *want_z = get(&G, "z");
    const Block *want_r = get(&G, "reward");

    const int deter = h0->cols, stoch = z0->cols, hidden = p0w->rows;
    const int batch = h0->rows, steps = acts->cols;
    const int act_dim = w_ih->cols - stoch;

    if (w_ih->rows != 3 * deter || w_hh->cols != deter || act_dim != 1) {
        fprintf(stderr, "unexpected shapes in the export\n");
        return 2;
    }
    if (want_h->cols != steps * deter || want_z->cols != steps * stoch
        || want_r->cols != steps || want_r->rows != batch) {
        fprintf(stderr, "rollout shapes disagree with h0/actions\n");
        return 2;
    }
    printf("deter %d stoch %d hidden %d, %d sequences of %d open loop steps\n",
           deter, stoch, hidden, batch, steps);

    double *h = malloc(sizeof(double) * deter);
    double *z = malloc(sizeof(double) * stoch);
    double *x = malloc(sizeof(double) * (stoch + act_dim));
    double *gi = malloc(sizeof(double) * 3 * deter);
    double *gh = malloc(sizeof(double) * 3 * deter);
    double *feat = malloc(sizeof(double) * (deter + stoch));
    double *hid = malloc(sizeof(double) * hidden);
    double *pri = malloc(sizeof(double) * p2w->rows);
    double rout;
    if (!h || !z || !x || !gi || !gh || !feat || !hid || !pri) return 2;

    double max_h = 0.0, max_z = 0.0, max_r = 0.0, max_std = 0.0;
    for (int s = 0; s < batch; s++) {
        for (int i = 0; i < deter; i++) h[i] = at(h0, s, i);
        for (int i = 0; i < stoch; i++) z[i] = at(z0, s, i);

        for (int k = 0; k < steps; k++) {
            for (int i = 0; i < stoch; i++) x[i] = z[i];
            x[stoch] = at(acts, s, k);

            affine(w_ih, b_ih, x, gi);
            affine(w_hh, b_hh, h, gh);
            for (int i = 0; i < deter; i++) {
                const double r = sigmoidd(gi[i] + gh[i]);
                const double zz = sigmoidd(gi[deter + i] + gh[deter + i]);
                const double n = tanh(gi[2 * deter + i] + r * gh[2 * deter + i]);
                h[i] = (1.0 - zz) * n + zz * h[i];
            }

            affine(p0w, p0b, h, hid);
            for (int i = 0; i < hidden; i++) hid[i] = elu(hid[i]);
            affine(p2w, p2b, hid, pri);
            for (int i = 0; i < stoch; i++) z[i] = pri[i];
            /* The standard deviation does not feed the mean rollout, but it is
             * half of what prior_net emits and is exercised here so a wrong
             * softplus cannot hide. */
            for (int i = 0; i < stoch; i++) {
                const double std = softplus(pri[stoch + i]) + 0.1;
                if (std <= 0.1 || !isfinite(std)) {
                    fprintf(stderr, "prior std %g is not above min_std\n", std);
                    return 1;
                }
                if (std > max_std) max_std = std;
            }

            for (int i = 0; i < deter; i++) feat[i] = h[i];
            for (int i = 0; i < stoch; i++) feat[deter + i] = z[i];
            affine(r0w, r0b, feat, hid);
            for (int i = 0; i < hidden; i++) hid[i] = elu(hid[i]);
            affine(r2w, r2b, hid, &rout);

            for (int i = 0; i < deter; i++) {
                const double d = fabs(h[i] - at(want_h, s, k * deter + i));
                if (d > max_h) max_h = d;
            }
            for (int i = 0; i < stoch; i++) {
                const double d = fabs(z[i] - at(want_z, s, k * stoch + i));
                if (d > max_z) max_z = d;
            }
            const double d = fabs(rout - at(want_r, s, k));
            if (d > max_r) max_r = d;
        }
    }

    printf("  largest |C - torch| over %d steps:\n", batch * steps);
    printf("    deterministic state h  %.3e\n", max_h);
    printf("    prior mean z           %.3e\n", max_z);
    printf("    predicted reward       %.3e\n", max_r);
    printf("  largest prior std seen   %.3f\n", max_std);

    if (max_h > TOL || max_z > TOL || max_r > TOL) {
        printf("\nFAIL: C does not reproduce the PyTorch rollout to %.0e\n", TOL);
        return 1;
    }
    printf("\nC reproduces the PyTorch open loop rollout to within %.0e\n", TOL);
    return 0;
}
