#!/usr/bin/env bash

for exp in colors0 colors1; do
    for arch in "-W 101" "-W 101 101"; do
        for lambda2 in 1 0.1 0; do
            for lambda1 in 1 0.1 0; do
                python main.py $exp full_full --passive -k 5 -p 0.2 -E 1000 \
                    $arch --lambda1 $lambda1 --lambda2 $lambda2 --use-corrections
            done
        done
    done
done
