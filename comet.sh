#!/bin/bash

languages="it fi es el lt cs da fr hu lv nl pl sk sl sv hr bg ro"
#de et ga pt mt
system="aya-23-8B.1shot"

prefix_ref="data_2023.smpl50"
mkdir -p "$prefix_ref"
for lang in $languages; do
    tail -50 "data_2023/ECB.$lang.txt" > ${prefix_ref}/ECB.$lang.txt
done
tail -50 "data_2023/ECB.en.txt" > ${prefix_ref}/ECB.en.txt

prefix_out="results.2023"
for lang in $languages; do
    #en2xx
    src_file="${prefix_ref}/ECB.en.txt"
    ref_file="${prefix_ref}/ECB.$lang.txt"
    hyp_file="${prefix_out}/$system.en2$lang.txt"
    comet-score -s ${src_file} -t ${hyp_file} -r ${ref_file} --model Unbabel/wmt22-cometkiwi-da --gpus 1 --only_system
    #xx2en
    hyp_file="${prefix_out}/$system.${lang}2en.txt"
    comet-score -s ${ref_file} -t ${hyp_file} -r ${src_file} --model Unbabel/wmt22-cometkiwi-da --gpus 1 --only_system
done