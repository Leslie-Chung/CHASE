SELECT qid, category, tid
FROM (
    SELECT laion100.sample_id as qid, laion1m.sample_id as tid, laion1m.nsfw as category, RANK() OVER (PARTITION BY laion100.sample_id, laion1m.nsfw ORDER BY laion1m.vec <#> laion100.vec) AS rank
    FROM laion1m join laion100 on laion1m.vec <#> laion100.vec < -0.83
) AS ranked
WHERE ranked.rank <= 10
ORDER BY qid;