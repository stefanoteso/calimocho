#!/usr/bin/env bash

for arch in "-W 3 3"; do
    for lambda2 in 1 0.1 0.01 0; do
        for lambda1 in 1 0.1 0.01 0; do
            python main.py xor full_full --passive -k 5 -p 0.2 -E 1000 \
                $arch --lambda1 $lambda1 --lambda2 $lambda2 --use-corrections
        done
    done
done
