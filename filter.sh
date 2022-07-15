#!/bin/bash

target_dir='/home/bzheng/workspace/debug/benchmark/nnc_summary/'
boundary_ops=('aten::addmm' 'aten::bmm' 'aten::conv_transpose1d' 'aten::conv_transpose2d' 'aten::conv1d' 'aten::conv2d' 'aten::conv3d' 'aten::linear' 'aten::matmul')

files=`ls $target_dir`
for file in $files
do
    filename=`echo $file | awk -F'.' '{printf $1}'`
    num=`echo $filename | awk -F'_' '{printf $3}'`
    if [ -f ${target_dir}/temp_$file ]; then
        continue
    fi
    while read line
    do
        if [[ $file == "temp"* ]]; then
            continue
        else
            if [[ $line == "Pattern"* ]]; then
                # echo $line' Normalized time' > ${target_dir}/temp_$file
                echo $line > ${target_dir}/temp_$file
            fi
            for boundary in "${boundary_ops[@]}"
            do
                if [[ $line == $boundary* ]]; then
                    match="0"
                    for b in "${boundary_ops[@]}"
                    do
                        if [[ $line == $boundary*$b* ]]; then
                            match="1"
                            break
                        fi
                    done
                    if [[ "$match" == "0" ]]; then
                        # time=`echo $line | awk -F' ' '{printf $3}'`
                        # modelnum=`echo $line | awk -F' ' '{printf $5}'`
                        # echo $line' '$(printf "%.10f" $(echo "scale=10;$time/$modelnum"|bc)) >> ${target_dir}/temp_$file
                        echo $line >> ${target_dir}/temp_$file
                    fi
                fi
            done
        fi
    done < $target_dir/$file
done

files=`ls $target_dir`
for file in $files
do
    if [[ ! $file == "temp"* ]]; then
        continue
    fi
    if [[ -f ${target_dir}/new_$file ]]; then
        continue
    fi
    filename=`echo $file | awk -F'.' '{printf $1}'`
    num=`echo $filename | awk -F'_' '{printf $4}'`
    if [[ $num == "1" ]]; then
        continue
    fi
    while read line
    do
        if [[ $line == "Pattern"* ]]; then
            echo $line' Perf_gain' > ${target_dir}/new_$file
            continue
        fi
        boundary=`echo $line | awk -F',' '{printf $1}'`
        while read l
        do
            if [[ $l == $boundary* ]]; then
                temp1=`echo $l | awk -F' ' '{printf $6}'`
                temp2=`echo $line | awk -F' ' '{printf $6}'`
                echo $line' '$(printf "%.16f" $(echo "scale=16;$temp2/$temp1"|bc)) >> ${target_dir}/new_$file
            fi
        done < $target_dir/temp_nnc_summary_1.log
    done < $target_dir/$file
done

