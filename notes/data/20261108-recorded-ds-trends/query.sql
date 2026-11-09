SELECT year, mracehisp AS race_code,
       LEAST(45, GREATEST(19, mager)) AS age_stratum,
       COUNT(*) AS births,
       COUNT(*) FILTER (WHERE COALESCE(ca_down,ca_downs)='C') AS confirmed,
       COUNT(*) FILTER (WHERE COALESCE(ca_down,ca_downs)='P') AS pending,
       COUNT(*) FILTER (WHERE COALESCE(ca_down,ca_downs)='N') AS no,
       COUNT(*) FILTER (WHERE COALESCE(ca_down,ca_downs)='U') AS unknown,
       COUNT(*) FILTER (WHERE COALESCE(ca_down,ca_downs) IS NULL) AS missing,
       COUNT(*) FILTER (WHERE mager IS NULL OR mager NOT BETWEEN 12 AND 50) AS invalid_age,
       COUNT(*) FILTER (WHERE mracehisp IS DISTINCT FROM
           CASE WHEN mhisp_r=9 THEN 8 WHEN mhisp_r BETWEEN 1 AND 5 THEN 7
                WHEN mhisp_r=0 THEN mrace6 END) AS race_mismatches,
       COUNT(*) FILTER (WHERE COALESCE(ca_down,ca_downs) IS DISTINCT FROM ca_down_c) AS status_mismatches
FROM us_births WHERE year BETWEEN 2018 AND 2025
GROUP BY year, mracehisp, LEAST(45, GREATEST(19, mager))
ORDER BY year, race_code, age_stratum
