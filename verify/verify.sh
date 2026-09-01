#!/usr/bin/env bash
# Recompute what this repository publishes, in languages that share no code with
# it, and require agreement.
#
# Every number in the README came out of one PyTorch program. The figures read
# the CSVs that program wrote, and scripts/check_numbers.py checks those CSVs
# against the README in Python again, so an error in the Python would be
# reproduced by everything that looks at it. These are independent
# implementations: a mistake would have to be made identically in several
# languages to survive.
#
# Each is skipped with a clear message if its toolchain is absent, so this runs
# on a laptop with only some of them. CI has all of them.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

pass=0 fail=0 skip=0

run () {
    local name="$1" tool="$2"; shift 2
    printf '\n=== %s ===\n' "$name"
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'skipped: %s is not installed\n' "$tool"
        skip=$((skip + 1)); return
    fi
    if "$@"; then pass=$((pass + 1)); else fail=$((fail + 1)); fi
}

# SQL has no assertion of its own, so the comparison happens here: every figure
# it recomputes has to appear in the README, bounded so that 0.0403 does not
# match 0.04031.
check_sql () {
    local line label value pattern missing=0 checked=0
    # sqlite3 reads stdin, which inside a script is the rest of the script, and
    # its CSV output is CRLF. Both of those have bitten this repo before.
    while IFS= read -r line; do
        line=${line//$'\r'/}
        line=${line//\"/}
        [ -n "$line" ] || continue
        label=${line%,*}
        value=${line##*,}
        pattern=$(printf '%s' "$value" | sed 's/\./\\./g')
        checked=$((checked + 1))
        if ! grep -qE "(^|[^0-9.])${pattern}([^0-9]|$)" README.md notes/METHODS.md; then
            printf '  %-46s should read %s, not found\n' "$label" "$value"
            missing=$((missing + 1))
        fi
    done < <(sqlite3 -init verify/tables.sql :memory: "" < /dev/null 2>/dev/null)

    if [ "$checked" -eq 0 ]; then
        echo "SQL produced nothing"
        return 1
    fi
    if [ "$missing" -gt 0 ]; then
        printf '\n%d of %d figures recomputed in SQL are not in the documents\n' \
               "$missing" "$checked"
        return 1
    fi
    printf 'SQL recomputed %d published figures from results/ and found every one\n' "$checked"
    return 0
}

check_c () {
    cc -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror \
       -o "${TMPDIR:-/tmp}/wmfs_rssm_step" verify/rssm_step.c -lm || return 1
    "${TMPDIR:-/tmp}/wmfs_rssm_step" "$root"
}

check_go () { ( cd verify/gocheck && go run . -root "$root" ); }

check_rust () { ( cd verify/pendulum && cargo run --release --quiet -- "$root" ); }

run "SQL, the published tables"        sqlite3 check_sql
run "C, the dynamics step"             cc      check_c
run "Go, structure and provenance"     go      check_go
run "R, exact permutation inference"   Rscript Rscript verify/verify.R "$root"
run "Rust, the environment itself"     cargo   check_rust
run "JavaScript, claims written in words" node    node verify/curves.js "$root"
run "Ruby, prose against the source"   ruby    ruby verify/degradation.rb "$root"

printf '\n%s\n' "----------------------------------------"
printf '%d passed, %d failed, %d skipped\n' "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ] || exit 1
[ "$pass" -gt 0 ] || { echo "nothing ran"; exit 1; }
