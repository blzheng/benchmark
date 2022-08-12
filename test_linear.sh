#!/bin/bash

submodule_dir=`pwd`/submodules

rm -rf linear*log

for len in `ls ${submodule_dir}`
do
	if [[ $len != "len1" && $len != "len2" ]]; then
		continue
	fi
	for pat in `ls ${submodule_dir}/$len`
	do
		if [[ $pat == "linear"* ]]; then
			while read line; do
				echo $line | tee -a ${pat}_40cores_prepacked_mkl.log 
				echo $line | tee -a ${pat}_40cores_unprepacked_mkl.log 
				echo $line | tee -a ${pat}_4cores_prepacked_mkl.log 
				echo $line | tee -a ${pat}_4cores_unprepacked_mkl.log 
				echo $line | tee -a ${pat}_40cores_onednn.log 
				echo $line | tee -a ${pat}_4cores_onednn.log 
				line=`echo $line | awk -F' ' '{printf $1}'`
				print="0"
				while read content; do
					if [[ $content == *"forward"* ]];
					then
						new_str=`echo ${content/def forward(self, /""}`
						new_str=`echo ${new_str/ /""}`
						new_str=`echo ${new_str/):/""}`
						print="0"
					elif [[ $content == *"super"* || $content == *"m = M()"* ]]; then
						print="1"
					elif [[ $content == *"start ="* ]]; then
						print="0"
					elif [[ $print == "1" && $content != "" ]]; then
						echo $content | tee -a ${pat}_40cores_prepacked_mkl.log 
						echo $content | tee -a ${pat}_4cores_prepacked_mkl.log 
						echo $content | tee -a ${pat}_40cores_unprepacked_mkl.log 
						echo $content | tee -a ${pat}_4cores_unprepacked_mkl.log 
						echo $content | tee -a ${pat}_40cores_onednn.log 
						echo $content | tee -a ${pat}_4cores_onednn.log 
					fi
				done < $line
				sed -i '2i\import intel_extension_for_pytorch as ipex' $line
				sed -i '/^start = time.time()/i m = ipex.optimize(m, dtype=torch.float32, auto_kernel_selection=True)' $line
				sed -i '/^start = time.time()/i with torch.no_grad():' $line
				sed -i '/^start = time.time()/i \ \ \ \ m = torch.jit.trace(m, ('${new_str}'))' $line
				sed -i '/^start = time.time()/i \ \ \ \ m = torch.jit.freeze(m)' $line
				sed -i '/^start = time.time()/i \ \ \ \ for i in range(30):' $line
				sed -i '/^start = time.time()/i \ \ \ \ \ \ \ \ output = m('${new_str}')' $line
				sed -i '/^start = time.time()/i \ \ \ \ total = 0' $line
				sed -i '/^start = time.time()/i \ \ \ \ for i in range(10):' $line
				sed -i 's/^start = time.time()/\ \ \ \ \ \ \ \ start = time.time()/g' $line
				sed -i 's/^output = /\ \ \ \ \ \ \ \ output = /g' $line
				sed -i 's/^end = /\ \ \ \ \ \ \ \ end = /g' $line
				sed -i '/^print/i \ \ \ \ \ \ \ \ total += end - start' $line
				sed -i 's/^print(end-start)/\ \ \ \ print(total)/g' $line
				python -m intel_extension_for_pytorch.cpu.launch --enable_jemalloc --node_id 0 $line | tee -a ${pat}_40cores_prepacked_mkl.log
				python -m intel_extension_for_pytorch.cpu.launch --enable_jemalloc --node_id 0 --ncore_per_instance 4 --ninstances 10 $line | tee -a ${pat}_4cores_prepacked_mkl.log
				sed -i 's/auto_kernel_selection=True/auto_kernel_selection=False/g' $line
				python -m intel_extension_for_pytorch.cpu.launch --enable_jemalloc --node_id 0 $line | tee -a ${pat}_40cores_unprepacked_mkl.log
				python -m intel_extension_for_pytorch.cpu.launch --enable_jemalloc --node_id 0 --ncore_per_instance 4 --ninstances 10 $line | tee -a ${pat}_4cores_unprepacked_mkl.log
				sed -i '/ipex.optimize/i ipex._enable_dnnl()' $line
				sed -i 's/auto_kernel_selection=False/auto_kernel_selection=True/g' $line
				python -m intel_extension_for_pytorch.cpu.launch --enable_jemalloc --node_id 0 $line | tee -a ${pat}_40cores_onednn.log
				python -m intel_extension_for_pytorch.cpu.launch --enable_jemalloc --node_id 0 --ncore_per_instance 4 --ninstances 10 $line | tee -a ${pat}_4cores_onednn.log
				sed -i '2d' $line
				sed -i '/ipex._enable_dnnl()/d' $line
				sed -i '/m = ipex.optimize*/d' $line
				sed -i '/with torch.no_grad()*/d' $line
				sed -i '/m = torch.jit.trace(m, */d' $line
				sed -i '/m = torch.jit.freeze(m)*/d' $line
				sed -i '/for i in range(*/d' $line
				sed -i 's/\ \ \ \ print(total)/print(end-start)/g' $line
				sed -i '/total\ */d' $line
				sed -i 's/\ \ \ \ \ \ \ \ start = time.time()/start = time.time()/g' $line
				sed -i 's/\ \ \ \ \ \ \ \ end = time.time()/end = time.time()/g' $line
				sed -i 's/\ \ \ \ \ \ \ \ output = m(/output = m(/g' $line
				sed -i '0,/output = m('${new_str}')*/{//d}' $line
			done < ${submodule_dir}/$len/$pat/all_deduplicated.txt
		fi
	done
done
