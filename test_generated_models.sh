#!/bin/bash
CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS - 1`
benchmark_dir=`pwd`

rm -rf ${benchmark_dir}/submodules/
mkdir -p ${benchmark_dir}/submodules/ 


python rewrite_models.py

dir="${benchmark_dir}/models/"
for item in `ls $dir`
do
    echo $item
    python $dir$item
done

dir="${benchmark_dir}/temp/"
result_dir="${benchmark_dir}/shapes/"
for item in `ls $dir`
do
    echo $item
    name=`echo $item | awk -F'.' '{printf $1}'`
    python $dir$item | tee ${result_dir}${name}_BS_3_224_224
done


for batch_size in 1 $(expr ${TOTAL_CORES} / 2) ${TOTAL_CORES} $(expr ${TOTAL_CORES} \* 2) $(expr ${TOTAL_CORES} \* 4)
do
    python auto_gen.py ${batch_size}
done

dir="${benchmark_dir}/submodules/"
for len in `ls $dir`
do
    for pattern in `ls $dir/$len`
    do
        timestamp=`date +%Y%m%d_%H%M%S`
        perf_file_name=${pattern}_perf_${timestamp}.txt        
        core_cnt=-1
        echo "${pattern}" | tee $perf_file_name
        for item in `ls $dir/$len/$pattern`
        do
            if [[ core_cnt -eq $TOTAL_CORES ]]; then
                core_cnt=-1
        	    wait
            fi
            if [[ $item == *".py" ]]; then
                ((core_cnt++)); numactl --physcpubind ${core_cnt} python $dir/$len/$pattern/$item | tee -a $perf_file_name &
            fi
        done
        wait
        if [ -f $dir/$len/$pattern/all.txt ];
        then
            continue
        fi
        python analyze_submodules.py --dir $dir/$len/$pattern
    done
done
