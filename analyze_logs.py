import torch
import torchvision
import geffnet
import os
import argparse
import sys
from utils import *
import re

parser = argparse.ArgumentParser(description='Analyze logs')
parser.add_argument('--log', type=str, help='log file')
parser.add_argument('--regenerate', type=bool, default=False, help='Whether to regenerate files with error')
args = parser.parse_args()

def get_model(name):
    try:
        model = torchvision.models.__dict__[name]()
    except KeyError as e:
        print("This model is not a torchvision model, try geffnet...")
        try:
            model = geffnet.create_model(name.split("_geffnet")[0], pretrained=True)
        except Exception as e2:
            print("This model is not a geffnet model")
            exit()
    return model

def regenerate(modelname, pattern, inputstr):
    model = get_model(modelname)
    oplists = pattern.split("_")
    inputs, module_dict, attr_dict, forward_list = parse_graph(model)
    module_dict, attr_dict, forward_list = generate_model_contents(module_dict, attr_dict, forward_list)
    shapes_dict = get_shapes(modelname, inputstr)
    forward_list = simplify_forward_list(forward_list)
    pattern_list = find_pattern(forward_list, oplists)
    for i in range(len(pattern_list)):
        sub_module_dict, sub_attr_dict = get_sub_module_attr_dict(pattern_list[i], module_dict, attr_dict)
        inputs, outputs = get_inputs_outputs(pattern_list[i])
        dirpath = os.getcwd() + "/submodules/len"+str(len(oplists))+"/"+get_oplists_str(oplists)+"/"
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        generate_file(dirpath+modelname+"_"+str(i)+".py", inputs, outputs, shapes_dict, sub_module_dict, sub_attr_dict, pattern_list[i])


logfile = args.log
gen = args.regenerate

error_files=[]
cur_file=""
cur_status=False
with open(logfile, 'r') as f:
    content = f.readlines()
    for line in content:
        if line.strip() == "":
            continue
        if line.startswith("/home/"):
            if not cur_status and cur_file != "":
                error_files.append(cur_file)
            cur_file = line
            cur_status = False
        elif re.match("^[0-9]+.[0-9]*$", line):
            cur_status = True

for err_file_name in error_files:
    if err_file_name.strip() == "":
        continue
    pattern = err_file_name.split("/")[-2]
    file_name = err_file_name.split("/")[-1]
    modelname = file_name.replace("_"+file_name.split("_")[-1], "")
    print(pattern, modelname)
    if gen:
        regenerate(modelname, pattern, "1_3_224_224")