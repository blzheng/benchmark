import torch
import torchvision.models as models
from utils import *
import os
from pretrained_models import pretrained_models

def auto_generate(modelname, pattern, inputstr):
    oplists = parse_pattern(pattern)
    model = pretrained_models[modelname]
    # model = models.inception_v3(pretrained = True)
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


# name = "inception_v3_google"
# pattern = "('aten::cat', )"
# auto_generate(name, pattern, "1_3_224_224")
# exit()
# name = "alexnet"
# pattern = "('aten::conv2d', 'aten::relu', )"
# pattern = "('aten::conv2d', )"
pattern_file = "patterns.txt"
with open(pattern_file, "r") as p_reader:
    patterns = p_reader.readlines()
    for pattern in patterns:
        # oplists = parse_pattern(pattern)
        for name in pretrained_models:
            print("============================")
            print(name, pattern)
            auto_generate(name, pattern, "1_3_224_224")
