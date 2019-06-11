#!/usr/bin/env bash

for arch in "-W 3" "-W 3 3"; do
    for lambdas in "1 0" "0.9 0" "0.9 0.1" "0.5 0" "0.5 0.25" "0.1 0" "0.1 0.01" "0 0" "0 0.01"; do
        python main.py xor full_full --passive -k 5 -p 0.2 \
            $arch -E 1000 --lambdas $lambdas
    done
done
