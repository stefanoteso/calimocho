#!/usr/bin/env bash

for exp in colors0 colors1; do
    for arch in "-W 101" "-W 101 101"; do
        for lambdas in "1 0" "0.9 0" "0.5 0" "0.1 0"; do
            python main.py $exp full_full --passive -k 5 -p 0.2 \
                $arch -E 1000 --lambdas $lambdas
        done
    done
done
