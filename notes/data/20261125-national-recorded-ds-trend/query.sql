WITH s AS (
    SELECT year, mage_c AS age, mager, down_ind, p_ds_lb_nt,
           COALESCE(ca_down, ca_downs, '') AS rv,
           COALESCE(uca_downs, downs, 0) AS od
    FROM us_births WHERE year BETWEEN 1989 AND 2025
)
SELECT year, age, CAST(rv <> '' AS INTEGER) AS revised,
       COUNT(*) AS births,
       COUNT(*) FILTER (WHERE rv = 'C') AS confirmed,
       COUNT(*) FILTER (WHERE rv = 'P') AS pending,
       COUNT(*) FILTER (WHERE rv = '' AND od = 1) AS unrevised_yes,
       COUNT(*) FILTER (WHERE rv = 'N' OR (rv = '' AND od = 2)) AS no,
       COUNT(*) FILTER (WHERE rv = 'U' OR (rv = '' AND od NOT IN (1, 2))) AS unknown,
       COUNT(*) FILTER (WHERE
           (rv IN ('C', 'P') OR (rv = '' AND od = 1)) <> COALESCE(down_ind = 1, FALSE)
           OR (rv = 'N' OR (rv = '' AND od = 2)) <> COALESCE(down_ind = 0, FALSE)
       ) AS status_mismatches,
       COUNT(*) FILTER (WHERE mager IS NOT NULL AND mager <> age) AS age_mismatches,
       SUM(p_ds_lb_nt) AS morris_expected_db
FROM s
GROUP BY year, age, revised
ORDER BY year, age, revised
