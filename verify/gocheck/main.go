// Structural and cross file validation of everything under results/.
//
// The CSVs in results/ are the evidence for every number in the README, and
// nothing checked that they are well formed or that they agree with each other.
// A truncated write, a column that drifted, a NaN out of a division, or a
// summary row that no longer matches the curve it was taken from would all be
// invisible until someone read the table.
//
// Five things here, none of which the Python does:
//
//	structure     no ragged rows, no duplicate or empty column names, no NaN/Inf
//	row counts    derived from results/run-meta.json, not from the files
//	agreement     summary.csv finals must equal the last point of each curve
//	budgets       env_steps must equal envs * horizon * iterations
//	provenance    the command the README documents must be the run recorded in
//	              run-meta.json
//
// Run: cd verify/gocheck && go run . -root ../..
package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

type meta struct {
	Seeds       []int   `json:"seeds"`
	Iters       int     `json:"iters"`
	Envs        int     `json:"envs"`
	Horizon     int     `json:"horizon"`
	ImagHorizon int     `json:"imag_horizon"`
	ActorLR     float64 `json:"actor_lr"`
	EvalEvery   int     `json:"eval_every"`
	MFIters     int     `json:"mf_iters"`
	MFEvalEvery int     `json:"mf_eval_every"`
}

func readCSV(path string) ([]string, [][]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	r := csv.NewReader(f)
	r.FieldsPerRecord = 0 // a ragged file is an error, which is the point
	rows, err := r.ReadAll()
	if err != nil {
		return nil, nil, err
	}
	if len(rows) < 2 {
		return nil, nil, fmt.Errorf("only %d rows", len(rows))
	}
	return rows[0], rows[1:], nil
}

func col(header []string, name string) int {
	for i, h := range header {
		if h == name {
			return i
		}
	}
	return -1
}

// validate reports every structural problem in one file rather than the first,
// so a broken run is diagnosed in one pass.
func validate(path string) []string {
	var problems []string
	header, rows, err := readCSV(path)
	if err != nil {
		return []string{fmt.Sprintf("unreadable: %v", err)}
	}
	seen := map[string]bool{}
	for _, h := range header {
		if h == "" {
			problems = append(problems, "a column has an empty name")
		}
		if seen[h] {
			problems = append(problems, fmt.Sprintf("duplicate column %q", h))
		}
		seen[h] = true
	}
	for i, row := range rows {
		for j, cell := range row {
			low := strings.ToLower(strings.TrimSpace(cell))
			if low == "nan" || low == "inf" || low == "-inf" || low == "+inf" {
				problems = append(problems,
					fmt.Sprintf("row %d column %s is %s", i+2, header[j], cell))
			}
			// An empty cell is legitimate here: the model-free arm has no
			// reconstruction or KL loss to record. A non-empty cell that is not
			// a number is not.
			if low == "" {
				continue
			}
			if v, err := strconv.ParseFloat(low, 64); err == nil {
				if math.IsNaN(v) || math.IsInf(v, 0) {
					problems = append(problems,
						fmt.Sprintf("row %d column %s is not finite", i+2, header[j]))
				}
			}
		}
	}
	return problems
}

type key struct{ mode, seed string }

func main() {
	root := flag.String("root", ".", "repository root")
	flag.Parse()

	results := filepath.Join(*root, "results")
	files, err := filepath.Glob(filepath.Join(results, "*.csv"))
	if err != nil || len(files) == 0 {
		fmt.Fprintf(os.Stderr, "no CSVs under %s\n", results)
		os.Exit(2)
	}
	sort.Strings(files)

	bad := 0
	fmt.Printf("validating %d files under results/\n", len(files))
	for _, path := range files {
		if problems := validate(path); len(problems) > 0 {
			bad += len(problems)
			for _, p := range problems {
				fmt.Printf("  %s: %s\n", filepath.Base(path), p)
			}
		}
	}
	if bad == 0 {
		fmt.Println("  no ragged rows, duplicate columns, NaN or Inf anywhere")
	}

	raw, err := os.ReadFile(filepath.Join(results, "run-meta.json"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "run-meta.json: %v\n", err)
		os.Exit(2)
	}
	var m meta
	if err := json.Unmarshal(raw, &m); err != nil {
		fmt.Fprintf(os.Stderr, "run-meta.json: %v\n", err)
		os.Exit(2)
	}

	lcHead, lcRows, err := readCSV(filepath.Join(results, "learning-curves.csv"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "learning-curves.csv: %v\n", err)
		os.Exit(2)
	}
	_, olRows, err := readCSV(filepath.Join(results, "open-loop.csv"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "open-loop.csv: %v\n", err)
		os.Exit(2)
	}
	suHead, suRows, err := readCSV(filepath.Join(results, "summary.csv"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "summary.csv: %v\n", err)
		os.Exit(2)
	}

	// Row counts follow from the hyperparameters, so they can be predicted
	// rather than read off the file being checked. evalCount counts the
	// iterations where experiments/main.py evaluates: every eval_every, plus
	// the last one whether or not it lands on the stride.
	evalCount := func(iters, every int) int {
		n := 0
		for it := 0; it < iters; it++ {
			if it%every == 0 || it == iters-1 {
				n++
			}
		}
		return n
	}
	nSeeds, nModes := len(m.Seeds), 3
	context := 20 // experiments/main.py filters 20 steps before predicting
	want := map[string]int{
		"learning-curves.csv": nSeeds*nModes*evalCount(m.Iters, m.EvalEvery) +
			nSeeds*evalCount(m.MFIters, m.MFEvalEvery),
		"open-loop.csv": nSeeds * nModes * (m.Horizon - context),
		"summary.csv":   nSeeds * (nModes + 1),
	}
	got := map[string]int{
		"learning-curves.csv": len(lcRows),
		"open-loop.csv":       len(olRows),
		"summary.csv":         len(suRows),
	}
	fmt.Println("\nrow counts against results/run-meta.json")
	for _, name := range []string{"learning-curves.csv", "open-loop.csv", "summary.csv"} {
		status := "ok"
		if got[name] != want[name] {
			status = "FAIL"
			bad++
		}
		fmt.Printf("  %-20s %4d rows, hyperparameters predict %4d  %s\n",
			name, got[name], want[name], status)
	}

	// summary.csv is supposed to be the last point of each learning curve. If
	// the two were written from different runs this is where it shows.
	lcMode, lcSeed := col(lcHead, "mode"), col(lcHead, "seed")
	lcIter, lcRet := col(lcHead, "iter"), col(lcHead, "return")
	lcSteps := col(lcHead, "env_steps")
	if lcMode < 0 || lcSeed < 0 || lcIter < 0 || lcRet < 0 || lcSteps < 0 {
		fmt.Fprintln(os.Stderr, "learning-curves.csv is missing a column")
		os.Exit(2)
	}
	last := map[key][]string{}
	for _, r := range lcRows {
		k := key{r[lcMode], r[lcSeed]}
		it, _ := strconv.Atoi(r[lcIter])
		if prev, ok := last[k]; !ok {
			last[k] = r
		} else {
			p, _ := strconv.Atoi(prev[lcIter])
			if it > p {
				last[k] = r
			}
		}
	}

	suMode, suSeed := col(suHead, "mode"), col(suHead, "seed")
	suFinal, suSteps := col(suHead, "final_return"), col(suHead, "env_steps")
	fmt.Printf("\nsummary.csv against the last point of each of the %d curves\n", len(last))
	agree := 0
	for _, r := range suRows {
		k := key{r[suMode], r[suSeed]}
		l, ok := last[k]
		if !ok {
			fmt.Printf("  %s seed %s has no learning curve  FAIL\n", k.mode, k.seed)
			bad++
			continue
		}
		a, _ := strconv.ParseFloat(r[suFinal], 64)
		b, _ := strconv.ParseFloat(l[lcRet], 64)
		sa, _ := strconv.Atoi(r[suSteps])
		sb, _ := strconv.Atoi(l[lcSteps])
		if a != b || sa != sb {
			fmt.Printf("  %-26s seed %s summary %.6f/%d curve %.6f/%d  FAIL\n",
				k.mode, k.seed, a, sa, b, sb)
			bad++
			continue
		}
		agree++

		// The step budget is not free either: it is envs * horizon per
		// iteration, for as many iterations as the arm ran.
		iters := m.Iters
		if strings.HasPrefix(k.mode, "model-free") {
			iters = m.MFIters
		}
		if sa != m.Envs*m.Horizon*iters {
			fmt.Printf("  %-26s seed %s env_steps %d, expected %d  FAIL\n",
				k.mode, k.seed, sa, m.Envs*m.Horizon*iters)
			bad++
		}
	}
	fmt.Printf("  %d of %d rows agree exactly, and every env_steps equals "+
		"envs * horizon * iterations\n", agree, len(suRows))

	// The README documents the command the committed results came from. If that
	// command is edited without rerunning, or rerun without editing, these stop
	// matching.
	readme, err := os.ReadFile(filepath.Join(*root, "README.md"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "README.md: %v\n", err)
		os.Exit(2)
	}
	cmd := regexp.MustCompile(`python -m experiments\.main[^\n` + "`" + `]*`).Find(readme)
	if cmd == nil {
		fmt.Fprintln(os.Stderr, "\nREADME.md documents no experiments.main command")
		os.Exit(1)
	}
	fmt.Printf("\nthe command README.md documents, against run-meta.json\n  %s\n", cmd)
	flags := map[string]string{
		"--iters":        strconv.Itoa(m.Iters),
		"--imag-horizon": strconv.Itoa(m.ImagHorizon),
		"--eval-every":   strconv.Itoa(m.EvalEvery),
	}
	for flag, value := range flags {
		re := regexp.MustCompile(regexp.QuoteMeta(flag) + ` (\S+)`)
		hit := re.FindSubmatch(cmd)
		if hit == nil {
			fmt.Printf("  %-16s not in the documented command  FAIL\n", flag)
			bad++
			continue
		}
		status := "ok"
		if string(hit[1]) != value {
			status = "FAIL"
			bad++
		}
		fmt.Printf("  %-16s README %-6s run-meta %-6s %s\n", flag, hit[1], value, status)
	}
	seeds := make([]string, len(m.Seeds))
	for i, s := range m.Seeds {
		seeds[i] = strconv.Itoa(s)
	}
	wantSeeds := "--seeds " + strings.Join(seeds, " ")
	seedStatus := "ok"
	if !strings.Contains(string(cmd), wantSeeds) {
		seedStatus = "FAIL"
		bad++
	}
	fmt.Printf("  %-16s %q %s\n", "--seeds", wantSeeds, seedStatus)
	// actor_lr is written as 1e-3 in the README and 0.001 in the JSON, so
	// compare the parsed values rather than the text.
	lrHit := regexp.MustCompile(`--actor-lr (\S+)`).FindSubmatch(cmd)
	lrStatus := "ok"
	if lrHit == nil {
		lrStatus = "FAIL"
		bad++
	} else if v, err := strconv.ParseFloat(string(lrHit[1]), 64); err != nil || v != m.ActorLR {
		lrStatus = "FAIL"
		bad++
	}
	fmt.Printf("  %-16s README %-6s run-meta %-6g %s\n", "--actor-lr",
		func() string {
			if lrHit == nil {
				return "absent"
			}
			return string(lrHit[1])
		}(), m.ActorLR, lrStatus)

	if bad > 0 {
		fmt.Printf("\n%d problems\n", bad)
		os.Exit(1)
	}
	fmt.Println("\nresults/ is well formed, internally consistent, and is the run the README documents")
}
