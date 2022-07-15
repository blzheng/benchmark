import torch
import torch.fx
import torchvision
from torch.fx.node import _format_arg
import builtins
import operator
import geffnet
import copy
import argparse
import sys
sys.path.append("..")
from utils import parse_graph, generate_model_contents, simplify_forward_list

parser = argparse.ArgumentParser(description='Rewrite models')
parser.add_argument('--model', type=str, help='model name')

# generate module
def rewrite_model(filename, inputs, module_dict, attr_dict, forward_list):
    modelname = filename.split("/")[-1].split(".py")[0]
    with open(filename, "w") as f:
        f.write("import torch\n")
        f.write("from torch import tensor\n")
        f.write("import torch.nn as nn\n")
        f.write("from torch.nn import *\n")
        f.write("import torchvision\n")
        f.write("import torchvision.models as models\n")
        f.write("from torchvision.ops.stochastic_depth import stochastic_depth\n")
        f.write("import time\n")
        f.write("import builtins\n")
        f.write("import operator\n\n")
        f.write("class M(torch.nn.Module):\n")
        f.write("    def __init__(self):\n")
        f.write("        super(M, self).__init__()\n")
        for key in module_dict.keys():
            f.write("        self."+key+" = "+str(module_dict[key]).replace("=none", "='none'")+"\n")
        for key in attr_dict.keys():
            f.write("        self."+key+" = "+str(attr_dict[key])+"\n")
        f.write("\n")
        inputstr=""
        for input in inputs:
            if not inputstr == "":
                inputstr = inputstr + ", " + input
            else:
                inputstr = input
        f.write("    def forward(self, "+inputstr+"):\n")
        for op in forward_list:
            if 'placeholder' in op:
                op = op.replace("placeholder(", "").replace(")", "")
            opstr = "        "+op+"\n"
            f.write(opstr.replace(",)\n", ")\n"))
        f.write("\n")
        f.write("m = M().eval()\n")
        # f.write("ref_m = "+pretrained_models_str[modelname]+".eval()\n")
        for input in inputs:
            f.write(input+" = torch.randn(1, 3, 224, 224)\n")
        # f.write("ref_output = ref_m("+inputstr+")\n")
        f.write("start = time.time()\n")
        f.write("output = m("+inputstr+")\n")
        f.write("end = time.time()\n")
        f.write("print(end-start)\n")
        # f.write("print(ref_output[0].shape==output[0].shape)\n")

def get_print_str(modelname, op):
    printstr = ""
    opret = op.split("=")[0]
    if not "shufflenet" in modelname:
        printstr += "        print('"+opret+": {}'.format("+opret+".shape))\n"
    else:
        printstr += "        if isinstance("+opret+", torch.Tensor):\n"
        printstr += "            print('"+opret+": {}'.format("+opret+".shape))\n"
        printstr += "        elif isinstance("+opret+", tuple):\n"
        printstr += "            tuple_shapes = '('\n"
        printstr += "            for item in "+opret+":\n"
        printstr += "               if isinstance(item, torch.Tensor):\n"
        printstr += "                   tuple_shapes += str(item.shape) + ', '\n"
        printstr += "               else:\n"
        printstr += "                   tuple_shapes += str(item) + ', '\n"
        printstr += "            tuple_shapes += ')'\n"
        printstr += "            print('"+opret+": {}'.format(tuple_shapes))\n"
        printstr += "        else:\n"
        printstr += "            print('"+opret+": {}'.format("+opret+"))\n"
    return printstr

# generate module with shapes
def rewrite_model_temp(filename, inputs, module_dict, attr_dict, forward_list):
    modelname = filename.split("/")[-1].split(".py")[0]
    with open(filename, "w") as f:
        f.write("import torch\n")
        f.write("from torch import tensor\n")
        f.write("import torch.nn as nn\n")
        f.write("from torch.nn import *\n")
        f.write("import torchvision\n")
        f.write("import torchvision.models as models\n")
        f.write("from torchvision.ops.stochastic_depth import stochastic_depth\n")
        f.write("import time\n")
        f.write("import builtins\n")
        f.write("import operator\n\n")
        f.write("class M(torch.nn.Module):\n")
        f.write("    def __init__(self):\n")
        f.write("        super(M, self).__init__()\n")
        for key in module_dict.keys():
            f.write("        self."+key+" = "+str(module_dict[key]).replace("=none", "='none'")+"\n")
        for key in attr_dict.keys():
            f.write("        self."+key+" = "+str(attr_dict[key])+"\n")
        f.write("\n")
        inputstr=""
        for input in inputs:
            if not inputstr == "":
                inputstr = inputstr + ", " + input
            else:
                inputstr = input
        f.write("    def forward(self, "+inputstr+"):\n")
        for op in forward_list:
            if 'placeholder' in op:
                op = op.replace("placeholder(", "").replace(")", "")
            opstr = "        "+op+"\n"
            f.write(opstr.replace(",)\n", ")\n"))
            if not "return " in op:
                f.write(get_print_str(modelname, op))
        f.write("\n")
        f.write("m = M().eval()\n")
        for input in inputs:
            f.write(input+" = torch.randn(1, 3, 224, 224)\n")
        f.write("output = m("+inputstr+")\n")


args = parser.parse_args()
name = args.model
model = torchvision.models.__dict__[name]()
inputs, module_dict, attr_dict, forward_list = parse_graph(model)
module_dict, attr_dict, forward_list = generate_model_contents(module_dict, attr_dict, forward_list)
forward_list = simplify_forward_list(forward_list)
rewrite_model("models/"+name+".py", inputs, module_dict, attr_dict, forward_list)
rewrite_model_temp("temp/"+name+".py", inputs, module_dict, attr_dict, forward_list)
