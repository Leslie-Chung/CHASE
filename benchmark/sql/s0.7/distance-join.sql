SELECT
    laion100.sample_id as qid, laion1M.sample_id as tid
FROM
   laion100
   JOIN laion1M ON laion1M.vec <#> laion100.vec < -0.83 AND laion1m.height > 220;