#!/usr/bin/env bash

# Offline learning
#for exp in colors0 colors1; do
#    for arch in "101" "101 101 101" "101 101 101 101 101"; do
#        for lambdas in "1 0" "0.9 0" "0.5 0" "0.1 0" "0 0"; do
#            python main.py $exp full_full --passive -k 5 -p 0.2 \
#                -W $arch -E 1000 --lambdas $lambdas -e 0.01
#        done
#    done
#done

# LIME stability experiment
#for exp in colors0 colors1; do
#    for arch in "101" "101 101 101" "101 101 101 101 101"; do
#        for lambdas in "0.9 0"; do
#            for limer in 5 10 25; do
#                for limes in 100 1000; do
#                    python main.py $exp full_full --passive -k 5 -p 0.2 \
#                        -W $arch -E 1000 --lambdas $lambdas -e 0.01 \
#                        --record-lime --lime-repeats $limer --lime-samples $limes
#                done
#            done
#        done
#    done
#done

# # Active learning
# for exp in colors0 colors1; do
#     for strategy in random margin; do
#         for arch in "101" "101 101 101" "101 101 101 101 101"; do
#             for lambdas in "1 0" "0.9 0" "0.9 0.1" "0.5 0" "0.5 0.1" "0.1 0" "0.1 0.1" "0 0" "0 0.1"; do
#                 python main.py $exp full_full \
#                     --strategy $strategy -k 5 -p 0.0001 -T 300 \
#                     -W $arch -E 100 --lambdas $lambdas -e 0.01
#             done
#
#             fmtarch=`echo $arch | tr ' ' ','`
#             python draw.py \
#                 $exp-active-W=$fmtarch-$strategy \
#                 results/${exp}__strategy=${strategy}__passive=False__*__W=${fmtarch}__*__L=*,0.0__*.pickle
#         done
#     done
# done

# Active learning from partial explanations
for exp in colors0; do
    for strategy in random; do
        for arch in "101"; do
            for lambdas in "1 0" "0 0"; do
                for prop_corr in 0.9 0.5 0.1; do
                    python main.py $exp full_full \
                        --strategy $strategy -k 5 -p 0.0001 -T 300 \
                        -W $arch -E 100 --lambdas $lambdas -e 0.01 \
                        -c $prop_corr
                done
            done
        done
    done
done
