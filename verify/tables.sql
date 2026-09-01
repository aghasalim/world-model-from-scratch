-- Recompute the two published tables from results/, in SQL.
--
-- The open loop table and the return table in the README are medians over three
-- seeds. They are produced by pandas in bench/figures.py and checked by
-- scripts/check_numbers.py, which is also Python. Both would agree with each
-- other about a wrong median. This derives the same figures from the row level
-- CSVs with nothing but SQLite, and verify/verify.sh looks for every value it
-- prints in the README.
--
-- Run: sqlite3 -init verify/tables.sql :memory: "" < /dev/null

.mode csv
.headers off
.import --csv results/open-loop.csv openloop
.import --csv results/summary.csv summary

-- SQLite has no median, so take the middle order statistic, averaging the two
-- middle values on an even count. That is the same definition as statistics.median.
CREATE TEMP VIEW ol_ranked AS
    SELECT mode, CAST(step AS INT) AS step, CAST(reward_mae AS REAL) AS mae,
           ROW_NUMBER() OVER (PARTITION BY mode, step ORDER BY CAST(reward_mae AS REAL)) AS rn,
           COUNT(*) OVER (PARTITION BY mode, step) AS n
    FROM openloop;

CREATE TEMP VIEW ol_median AS
    SELECT mode, step, AVG(mae) AS median
    FROM ol_ranked
    WHERE rn IN ((n + 1) / 2, (n + 2) / 2)
    GROUP BY mode, step;

CREATE TEMP VIEW ret_ranked AS
    SELECT mode, CAST(final_return AS REAL) AS ret,
           ROW_NUMBER() OVER (PARTITION BY mode ORDER BY CAST(final_return AS REAL)) AS rn,
           COUNT(*) OVER (PARTITION BY mode) AS n
    FROM summary;

-- The README writes returns as magnitudes after a minus sign, so the printed
-- form drops the sign the same way.
SELECT 'open-loop ' || mode || ' k=' || step || ',' || printf('%.4f', median)
FROM ol_median
WHERE step IN (1, 5, 10, 15, 20, 40)
ORDER BY mode, step;

SELECT 'return median ' || mode || ',' ||
       printf('%.2f', ABS((SELECT AVG(ret) FROM ret_ranked r2
                           WHERE r2.mode = r.mode AND r2.rn IN ((r2.n + 1) / 2, (r2.n + 2) / 2))))
FROM ret_ranked r GROUP BY mode ORDER BY mode;

SELECT 'return worst ' || mode || ',' || printf('%.1f', ABS(MIN(ret)))
FROM ret_ranked GROUP BY mode ORDER BY mode;

SELECT 'return best ' || mode || ',' || printf('%.1f', ABS(MAX(ret)))
FROM ret_ranked GROUP BY mode ORDER BY mode;

-- Seed level values the README argues from directly: the three k=1 errors for
-- each of the two arms whose bands it claims do not overlap.
SELECT 'seed ' || mode || ' k=1,' || printf('%.4f', mae)
FROM ol_ranked
WHERE step = 1 AND mode IN ('recon (Dreamer style)', 'no-recon (MuZero style)')
ORDER BY mode, mae;
