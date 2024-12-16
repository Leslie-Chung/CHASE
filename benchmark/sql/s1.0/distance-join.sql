SELECT
    laion100.sample_id as qid, laion1m.sample_id as tid
FROM
   laion100
   JOIN laion1m ON laion1m.vec <#> laion100.vec < -0.83;