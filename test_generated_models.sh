#!/bin/bash
dir="/home/bzheng/workspace/debug/benchmark/models/"
for item in `ls $dir`
do
    echo $item
    python $dir$item
done