#!/usr/bin/env bash

for exp in colors0 colors1; do
    for arch in "101" "101 101" "101 101 101" "101 101 101 101" " 101 101 101 101 101"; do
        for lambdas in "1 0" "0.9 0" "0.5 0" "0.1 0" "0 0"; do
            python main.py $exp full_full --passive -k 5 -p 0.2 \
                -W $arch -E 1000 --lambdas $lambdas -e 0.01
        done
    done
done
