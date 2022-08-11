import torch
import torch.fx
import torchvision.models as models
from torch.fx.node import _format_arg
import builtins
import operator
import geffnet
import copy
from pretrained_models import pretrained_models, pretrained_models_str
from utils import *
from models import *

# generate module
def rewrite_model(filename, inputs_dict, module_dict, attr_dict, forward_list):
    modelname = filename.split("/")[-1].split(".py")[0]
    inputs = inputs_dict.keys()
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
        f.write("import operator\n")
        f.write("import sys\n")
        f.write("import os\n\n")
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
        f.write("CORES=os.popen(\"lscpu | grep Core | awk '{print $4}'\").readlines()\n")
        f.write("SOCKETS=os.popen(\"lscpu | grep Socket | awk '{print $2}'\").readlines()\n")
        f.write("BS=int(CORES[0])*int(SOCKETS[0])\n")
        f.write("batch_size=BS\n")
        for k in inputs_dict:
            f.write(k+" = " + inputs_dict[k] + "\n")
        f.write("def print_throughput(flag):\n")
        f.write("    start_time=time.time()\n")
        f.write("    for i in range(10):\n")
        f.write("        output = m("+inputstr+")\n")
        f.write("    total_iter_time = time.time() - start_time\n")
        f.write("    Throughput = batch_size * 10 / total_iter_time\n")
        f.write("    file_current = os.path.basename(__file__)\n")
        f.write("    print(file_current,',',BS,',',flag,',',Throughput)\n")

        f.write("for flag in {False,True}:\n")
        f.write("    torch._C._jit_set_texpr_fuser_enabled(flag)\n")
        f.write("    print_throughput(flag)\n")

def get_print_str(modelname, op):
    printstr = ""
    opret = op.split("=")[0]
    if not "shufflenet" and not "+" in modelname:
        printstr += "        print('"+opret+": {}'.format("+opret+".shape))\n"
    else:
        printstr += "        if "+opret+" is None:\n"
        printstr += "            print('"+opret+": {}'.format("+opret+"))\n"
        printstr += "        elif isinstance("+opret+", torch.Tensor):\n"
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
def rewrite_model_temp(filename, inputs_dict, module_dict, attr_dict, forward_list):
    modelname = filename.split("/")[-1].split(".py")[0]
    inputs = inputs_dict.keys()
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
        f.write("import operator\n")
        f.write("import sys\n")
        f.write("import os\n\n")
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
        f.write("CORES=os.popen(\"lscpu | grep Core | awk '{print $4}'\").readlines()\n")
        f.write("SOCKETS=os.popen(\"lscpu | grep Socket | awk '{print $2}'\").readlines()\n")
        f.write("BS=int(CORES[0])*int(SOCKETS[0])\n")
        f.write("batch_size=BS\n")
        for k in inputs_dict:
            f.write(k + " = " + inputs_dict[k] + "\n")
        f.write("def print_throughput(flag):\n")
        f.write("    start_time=time.time()\n")
        f.write("    for i in range(10):\n")
        f.write("        output = m("+inputstr+")\n")
        f.write("    total_iter_time = time.time() - start_time\n")
        f.write("    Throughput = batch_size * 10 / total_iter_time\n")
        f.write("    file_current = os.path.basename(__file__)\n")
        f.write("    print(file_current,',',BS,',',flag,',',Throughput)\n")

        f.write("for flag in {False,True}:\n")
        f.write("    torch._C._jit_set_texpr_fuser_enabled(flag)\n")
        f.write("    print_throughput(flag)\n")

for name in pretrained_models.keys():
    model = pretrained_models[name]
    inputs, module_dict, attr_dict, forward_list = parse_graph(model)
    module_dict, attr_dict, forward_list = generate_model_contents(module_dict, attr_dict, forward_list)
    forward_list = simplify_forward_list(forward_list)
    inputs_dict = get_inputs_dict(name, inputs)
    rewrite_model("models/"+name+".py", inputs_dict, module_dict, attr_dict, forward_list)
    rewrite_model_temp("temp/"+name+".py", inputs_dict, module_dict, attr_dict, forward_list)
