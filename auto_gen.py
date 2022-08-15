import imp
import torch
import torchvision.models as models
from utils import *
import os
from pretrained_models import pretrained_models

import time
import multiprocessing
import sys

pattern_file = "patterns.txt"
start = False
nnc_bm_flag=True
pool = multiprocessing.Pool(processes = 80)
for name in pretrained_models:
    model = pretrained_models[name]
    inputs, module_dict, attr_dict, forward_list = parse_graph(model)
    module_dict, attr_dict, forward_list = generate_model_contents(module_dict, attr_dict, forward_list)
    shapes_dict = get_shapes(name, "1_3_224_224")
    forward_list = simplify_forward_list(forward_list)
    with open(pattern_file, "r") as p_reader:
        patterns = p_reader.readlines()
        for pattern in patterns:
            print("============================")
            print(name, pattern)
            oplists = parse_pattern(pattern)
            pattern_list = find_pattern(forward_list, oplists)
            for i in range(len(pattern_list)):
                sub_module_dict, sub_attr_dict = get_sub_module_attr_dict(pattern_list[i], module_dict, attr_dict)
                inputs, outputs = get_inputs_outputs(pattern_list[i])
                dirpath = os.getcwd() + "/submodules/len"+str(len(oplists))+"/"+get_oplists_str(oplists)+"/"
                if not os.path.exists(dirpath):
                    os.makedirs(dirpath)
                #generate_file(dirpath+name+"_"+str(i)+".py", inputs, outputs, shapes_dict, sub_module_dict, sub_attr_dict, pattern_list[i])
                pool.apply_async(generate_file,(nnc_bm_flag,sys.argv[1],dirpath+name+"_"+sys.argv[1]+"_"+str(i)+".py", inputs, outputs, shapes_dict, sub_module_dict, sub_attr_dict, pattern_list[i]))
pool.close()
pool.join()