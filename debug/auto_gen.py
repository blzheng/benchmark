import torch
import os
import argparse
import sys
sys.path.append("..")
from utils import *
from models import get_model

parser = argparse.ArgumentParser(description='Auto generate models')
parser.add_argument('--model', type=str, help='model name')
parser.add_argument('--pattern', type=str, help='pattern string')

def auto_generate(modelname, model, pattern, inputstr):
    oplists = parse_pattern(pattern)
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
        generate_file(False, "1",dirpath+modelname+"_1_"+str(i)+".py", inputs, outputs, shapes_dict, sub_module_dict, sub_attr_dict, pattern_list[i])


args = parser.parse_args()
name = args.model
pattern = args.pattern
model = get_model(name)
auto_generate(name, model, pattern, "1_3_224_224")
