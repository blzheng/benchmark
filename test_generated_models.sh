#!/bin/bash

python rewrite_models.py

dir="/home/bzheng/workspace/debug/benchmark/models/"
for item in `ls $dir`
do
    echo $item
    python $dir$item
done

dir="/home/bzheng/workspace/debug/benchmark/temp/"
result_dir="/home/bzheng/workspace/debug/benchmark/shapes/"
for item in `ls $dir`
do
    echo $item
    name=`echo $item | awk -F'.' '{printf $1}'`
    python $dir$item | tee ${result_dir}${name}_1_3_224_224
done

python auto_gen.py
# dir="/home/bzheng/workspace/debug/benchmark/submodules/len2/conv2d_relu/"
# for item in `ls $dir`
# do
#     echo $item
#     python $dir$item
# done