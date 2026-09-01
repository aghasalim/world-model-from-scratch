// Two claims the README makes in words rather than in a table.
//
// scripts/check_numbers.py says so itself: it checks quoted figures against
// results/, not claims written in words. Two of the claims written in words are
// load bearing.
//
// The first is "reward prediction loss falls by 23 times for recon, 23 times for
// no-recon and 35 times for contrastive", which is the whole evidence for the
// half of the repository that did work.
//
// The second is the environment step budget the model-free baseline is compared
// at. The model-free arm is evaluated every ten iterations and never lands on
// 57,600 steps, so the table quotes its nearest evaluation, and the README has
// to say which one that is rather than borrowing the world model's number.
//
// Run: node verify/curves.js .

const fs = require("fs");
const path = require("path");

const root = process.argv[2] || ".";
let failures = 0;

function readCsv(file) {
    const text = fs.readFileSync(file, "utf8").trim();
    const lines = text.split(/\r?\n/);
    const header = lines[0].split(",");
    return lines.slice(1).map((line) => {
        const cells = line.split(",");
        if (cells.length !== header.length) {
            throw new Error(`${path.basename(file)}: ragged row: ${line}`);
        }
        return Object.fromEntries(header.map((h, i) => [h, cells[i]]));
    });
}

function median(xs) {
    const s = [...xs].sort((a, b) => a - b);
    const m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

const curves = readCsv(path.join(root, "results", "learning-curves.csv"));
const summary = readCsv(path.join(root, "results", "summary.csv"));
// Collapse the line wrapping so a claim can be matched as one sentence.
const readme = fs.readFileSync(path.join(root, "README.md"), "utf8")
    .replace(/\*\*/g, "")
    .replace(/\s+/g, " ");

// ---- reward prediction loss, first evaluation against last ----------------
const MODES = [
    ["recon", "recon (Dreamer style)"],
    ["no-recon", "no-recon (MuZero style)"],
    ["contrastive", "contrastive"],
];

const ratios = {};
console.log("reward prediction loss, median first iteration over median last");
for (const [short, mode] of MODES) {
    const rows = curves.filter((r) => r.mode === mode && r.reward !== "");
    const iters = rows.map((r) => Number(r.iter));
    const first = Math.min(...iters);
    const last = Math.max(...iters);
    const f = median(rows.filter((r) => Number(r.iter) === first).map((r) => Number(r.reward)));
    const l = median(rows.filter((r) => Number(r.iter) === last).map((r) => Number(r.reward)));
    ratios[short] = f / l;
    console.log(`  ${short.padEnd(12)} iter ${first} ${f.toExponential(4)} -> ` +
                `iter ${last} ${l.toExponential(4)}  ${(f / l).toFixed(2)} times`);
}

const sentence = readme.match(
    /reward prediction loss falls by (\d+) times for recon, (\d+) times for no-recon and (\d+) times for contrastive/);
if (!sentence) {
    console.log("  FAIL: README no longer makes the loss reduction claim in that form");
    failures++;
} else {
    for (const [i, [short]] of MODES.entries()) {
        const claimed = Number(sentence[i + 1]);
        const got = Math.round(ratios[short]);
        const ok = claimed === got;
        if (!ok) failures++;
        console.log(`  README says ${claimed} times for ${short}, results/ give ` +
                    `${ratios[short].toFixed(2)} which rounds to ${got}  ${ok ? "ok" : "FAIL"}`);
    }
}

// ---- the budget the model-free baseline is quoted at ----------------------
const MF = "model-free (recurrent PG)";
const wmBudget = Math.max(...summary
    .filter((r) => r.mode !== MF)
    .map((r) => Number(r.env_steps)));

const mfSteps = [...new Set(curves.filter((r) => r.mode === MF)
    .map((r) => Number(r.env_steps)))].sort((a, b) => a - b);
const below = Math.max(...mfSteps.filter((s) => s <= wmBudget));
const above = Math.min(...mfSteps.filter((s) => s > wmBudget));

function mfMedian(steps) {
    return median(curves.filter((r) => r.mode === MF && Number(r.env_steps) === steps)
        .map((r) => Number(r.return)));
}

console.log(`\nthe world models get ${wmBudget} environment steps; the model-free arm is`);
console.log(`evaluated at ${below} and then at ${above}, never at ${wmBudget}`);

const claims = [
    [below, mfMedian(below)],
    [above, mfMedian(above)],
];
for (const [steps, value] of claims) {
    const withCommas = steps.toLocaleString("en-US");
    const shown = Math.abs(value).toFixed(2);
    const hasSteps = readme.includes(withCommas) || readme.includes(String(steps));
    const hasValue = new RegExp(`(?<![\\d.])${shown.replace(".", "\\.")}(?!\\d)`).test(readme);
    const ok = hasSteps && hasValue;
    if (!ok) failures++;
    console.log(`  ${withCommas} steps, median ${value.toFixed(4)}: README mentions the ` +
                `budget ${hasSteps}, the value ${hasValue}  ${ok ? "ok" : "FAIL"}`);
}

if (failures > 0) {
    console.log(`\n${failures} checks failed`);
    process.exit(1);
}
console.log("\nJavaScript reproduces the loss reduction claim and the step budget the " +
            "model-free row is quoted at");
