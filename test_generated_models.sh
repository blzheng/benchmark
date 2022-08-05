#!/bin/bash
cur_dir=`pwd`
mkdir -p ${cur_dir}/models/
mkdir -p ${cur_dir}/temp/
mkdir -p ${cur_dir}/shapes/
mkdir -p ${cur_dir}/submodules/
python rewrite_models.py

dir=${cur_dir}/models/
for item in `ls $dir`
do
    echo $item
    python $dir$item
done

dir=${cur_dir}/temp/
result_dir=${cur_dir}/shapes/
for item in `ls $dir`
do
    echo $item
    name=`echo $item | awk -F'.' '{printf $1}'`
    python $dir$item | tee ${result_dir}${name}_1_3_224_224
done

python auto_gen.py
dir=${cur_dir}/submodules/
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
        # if [ -f $dir/$len/$pattern/all.txt ];
        # then
        #     continue
        # fi
        python analyze_submodules.py --dir $dir/$len/$pattern
    done
done
