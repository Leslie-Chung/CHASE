SELECT qid, tid
FROM (
    SELECT laion100.sample_id AS qid, laion1m.sample_id AS tid, RANK() OVER (PARTITION BY laion100.sample_id ORDER BY laion1m.vec <#> laion100.vec) AS rank
    FROM laion1m, laion100 WHERE laion1m.height > 1193
) AS ranked
WHERE ranked.rank <= 50;