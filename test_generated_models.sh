#!/bin/bash

python rewrite_models.py

dir="/home2/yudongsi/benchmark/models/"
for item in `ls $dir`
do
    echo $item
    python $dir$item
done

dir="/home2/yudongsi/benchmark/temp/"
result_dir="/home2/yudongsi/benchmark/shapes/"
for item in `ls $dir`
do
    echo $item
    name=`echo $item | awk -F'.' '{printf $1}'`
    python $dir$item | tee ${result_dir}${name}_1_3_224_224
done

python auto_gen.py
dir="/home2/yudongsi/benchmark/submodules/"
for len in `ls $dir`
do
    for pattern in `ls $dir/$len`
    do
        for item in `ls $dir/$len/$pattern`
        do
            if [[ $item == *".py" ]]; then
                echo $item
                python $dir/$len/$pattern/$item
            fi
        done
        if [ -f $dir/$len/$pattern/all.txt ];
        then
            continue
        fi
        python analyze_submodules.py --dir $dir/$len/$pattern
    done
done