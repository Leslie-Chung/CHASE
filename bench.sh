#!/bin/bash

selectivities=(1.0 0.9 0.7 0.5 0.3 0.03)
sql_types=("topk" "range" "category" "distance-join" "category-join" "knn-join")
dbnames=("lingodb" "chase" "vbase" "pgvector" "pase")

run_benchmarks() {
    for selectivity in "${selectivities[@]}"; do
        for sql_type in "${sql_types[@]}"; do
            time=6
            if [ $sql_type = 'knn-join' ]; then
                time=1
            fi
            tmpvenv/bin/python3 benchmark/run.py --sql_type=${sql_type} --selectivity=${selectivity} --dbnames ${dbnames[@]} --times=1
                # echo selectivity=${selectivity}
                # cat /home/postgres/lingo-db/chase-perf.txt >> /home/postgres/lingo-db/benchmark/chase-perf.log
                # echo selectivity=${selectivity}
                # cat /tmp/vectordb/vbase-perf.txt >> /home/postgres/lingo-db/benchmark/vbase-perf.log
        done
    done
}

run_benchmarks


