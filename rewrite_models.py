import torch
import torch.fx
import torchvision.models as models
from torch.fx.node import _format_arg
import builtins
import operator
import geffnet
import copy
from pretrained_models import pretrained_models, pretrained_models_str

def fetch_attr(target, module):
    target_atoms = target.split('.')
    attr_itr = module
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}")
        attr_itr = getattr(attr_itr, atom)
    return attr_itr

def parse_graph(m):
    g = torch.fx.Tracer().trace(m.eval())
    inputs=[]
    module_dict={}
    attr_dict={}
    forward_list=[]
    # print(m)
    # print(g)
    for node in g.nodes:
        if node.op == 'call_module':
            submodule = fetch_attr(node.target, m)
            module_dict[node.target] = submodule
            opstr = node.name+"="+node.target+_format_arg(node.args)+_format_arg(node.kwargs)
            forward_list.append(opstr)
        elif node.op == 'call_function':
            func=node.op        
            opstr = node.name+"="+node._pretty_print_target(node.target)+_format_arg(node.args)+_format_arg(node.kwargs)
            forward_list.append(opstr)
        elif node.op == 'call_method':
            opstr = node.name+"=method%"+str(node.args[0])+"."+node.target+_format_arg(node.args[1:])+_format_arg(node.kwargs)
            forward_list.append(opstr)
        elif node.op == 'output':
            forward_list.append("return "+_format_arg(node.all_input_nodes))
            print(node)
        elif node.op == 'placeholder':
            forward_list.append(node.name+"=placeholder("+node.target+")")
            inputs.append(node.target)
            print(node.op, node.target, node.args)
        elif node.op == 'get_attr':
            value = m.state_dict()[node.target]
            attr_dict[node.target] = 'torch.rand('+str(value.shape)+')'
            forward_list.append(node.name+"=attr:"+node.target)
        else:
            print(node.op, node.target, node.args)
    return inputs, module_dict, attr_dict, forward_list

def generate_model_contents(module_dict, attr_dict, forward_list):
    new_module_dict={}
    new_attr_dict={}
    new_forward_list=copy.deepcopy(forward_list)
    puncs = [",", ")", "]", "=", "."]
    for i in range(len(forward_list)):
        # rename op
        if not "return" in forward_list[i] and not "placeholder" in forward_list[i] and not "attr:" in forward_list[i]:
            op = forward_list[i].split('=')[1].split('(')[0]
            if not "torch" in op and not "operator" in op and not "builtins" in op and not "method" in op:
                op_real = str(module_dict[op])
                op_name = op_real.split("(")[0].split(".")[-1].lower()
                idx = 0
                while True:
                    if op_name+str(idx) in new_module_dict:
                        idx+=1
                    else:
                        new_module_dict[op_name+str(idx)] = module_dict[op]
                        for j in range(len(forward_list)):
                            new_forward_list[j] = new_forward_list[j].replace("="+op+"(", "=self."+op_name+str(idx)+"(")
                        break
            else:
                # special cases
                if "builtins" in op and "shape" in new_forward_list[i]:
                    new_forward_list[i] = new_forward_list[i].replace("shape", "'shape'")
                if "bilinear" in forward_list[i]:
                    new_forward_list[i] = new_forward_list[i].replace("bilinear", "'bilinear'")
                elif "torchvision.ops.stochastic_depth.stochastic_depth" in new_forward_list[i]:
                    new_forward_list[i] = new_forward_list[i].replace("torchvision.ops.stochastic_depth.stochastic_depth", "stochastic_depth")
                    if "row" in new_forward_list[i]:
                        new_forward_list[i] = new_forward_list[i].replace("row", "'row'")
        elif "attr:" in forward_list[i]:
            name = forward_list[i].split("attr:")[-1]
            val_real = attr_dict[name]
            attr_name = name.split("(")[0].split(".")[-1].lower()
            idx = 0
            while True:
                if attr_name+str(idx) in new_attr_dict:
                    idx+=1
                else:
                    new_attr_dict[attr_name+str(idx)] = attr_dict[name]
                    for j in range(len(forward_list)):
                        new_forward_list[j] = new_forward_list[j].replace("=attr:"+name, "=self."+attr_name+str(idx))
                    break

        # rename input and output 
        out = new_forward_list[i].split('=')[0]
        new_out = "x" + str(i)
        new_forward_list[i] = new_forward_list[i].replace(out+"=", new_out+"=")
        for j in range(len(new_forward_list)):
            new_forward_list[j] = new_forward_list[j].replace(": ", "=")
            for p in puncs:
                new_forward_list[j] = new_forward_list[j].replace("%"+out+p, new_out+p)
            new_forward_list[j] = new_forward_list[j].replace(",){", ",").replace("){", ",").replace("}", ")").replace(",,", ",").replace("(,", "(")
            new_forward_list[j] = new_forward_list[j].replace("=method", "=")
    print(new_module_dict)
    print(new_attr_dict)
    print(new_forward_list)
    return new_module_dict, new_attr_dict, new_forward_list

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

for name in pretrained_models.keys():
    model = pretrained_models[name]
    inputs, module_dict, attr_dict, forward_list = parse_graph(model)
    module_dict, attr_dict, forward_list = generate_model_contents(module_dict, attr_dict, forward_list)
    rewrite_model("models/"+name+".py", inputs, module_dict, attr_dict, forward_list)
