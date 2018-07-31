#!/bin/bash

declare -a batches=("8" "16" "32" "64" "128")
declare -a networks=("resnet50" "resnet101" "resnet152")
# declare -a networks=("resnet101" "resnet152")
for n in "${networks[@]}"
do
for i in "${batches[@]}"
do 
    ./fused_resnet "${n}" "$i" 2>&1 | tee "fused_${n}_$i.log"
    mv "${n}.tsv" "${n}_${i}.tsv"
done
done

