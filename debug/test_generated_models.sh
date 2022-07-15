#!/bin/bash
export BENCHMAKR_DIR="/home/bzheng/workspace/debug/benchmark/debug/"
rm -rf ${BENCHMAKR_DIR}models/
rm -rf ${BENCHMAKR_DIR}temp/
rm -rf ${BENCHMAKR_DIR}shapes/
rm -rf ${BENCHMAKR_DIR}submodules/
mkdir -p ${BENCHMAKR_DIR}models/
mkdir -p ${BENCHMAKR_DIR}temp/
mkdir -p ${BENCHMAKR_DIR}shapes/
mkdir -p ${BENCHMAKR_DIR}submodules/

python rewrite_models.py --model "alexnet"

dir="${BENCHMAKR_DIR}models/"
for item in `ls $dir`
do
    echo $item
    python $dir$item
done

dir="${BENCHMAKR_DIR}temp/"
result_dir="${BENCHMAKR_DIR}shapes/"
for item in `ls $dir`
do
    echo $item
    name=`echo $item | awk -F'.' '{printf $1}'`
    python $dir$item | tee ${result_dir}${name}_1_3_224_224
done

python auto_gen.py --model "alexnet" --pattern "('aten::conv2d', 'aten::relu', )"
dir="${BENCHMAKR_DIR}submodules/"
len_dir=`ls $dir`
for ld in ${len_dir}
do
    ldir="$dir$ld/"
    pat_dir=`ls $ldir`
    for pd in ${pat_dir}
    do
        cur_dir="${ldir}${pd}/"
        files=`ls $cur_dir`
        for f in $files
        do
            cur_file="$cur_dir$f"
            echo $cur_file
            python $cur_file
        done
    done
done