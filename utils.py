import torch
import torch.fx
import torchvision.models as models
from torch.fx.node import _format_arg
import builtins
import operator
import geffnet
import copy
import glob

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

def parse_pattern(pattern):
    oplist = pattern.split("(")[-1].split(")")[0].split(",")
    new_oplist = []
    for i in range(len(oplist)):
        oplist[i] = oplist[i].replace("'", "").split(":")[-1].strip()
        if oplist[i] != "":
            new_oplist.append(oplist[i])
    return new_oplist

def get_shapes(modelname, shapestr):
    files = glob.glob("shapes/"+modelname+"_"+shapestr)
    shapes_dict = {}
    for f in files:
        with open(f, "r") as reader:
            contents = reader.readlines()
            for line in contents:
                var = line.split(": ")[0]
                shape = line.split(": ")[1].strip()
                shapes_dict[var] = shape
    return shapes_dict

def is_target(target, op):
    real = op.split("(")[0].split("=")[-1].strip()
    if "." in real:
        real = real.split(".")[-1]
    if real.startswith(target):
        return True
    return False

# find pattern
def find_pattern(forward_list, oplists):
    pattern_list = []
    for i in range(len(forward_list)):
        if is_target(oplists[0].lower(), forward_list[i]):
            is_match = True
            tmp = [forward_list[i]]
            last_output = forward_list[i].split("=")[0].strip()
            for j in range(1, len(oplists)):
                found = False
                for k in range(i+1, len(forward_list)):
                    if is_target(oplists[j].lower(), forward_list[k]) and last_output+"," in forward_list[k]:
                        tmp.append(forward_list[k])
                        last_output = forward_list[k].split("=")[0].strip()
                        found = True
                if not found or len(tmp) != j+1:
                    is_match = False
                    break
            if is_match:
                pattern_list.append(tmp)
    print(pattern_list)
    return pattern_list

# get module_dict, attr_dict for sub model
def get_sub_module_attr_dict(forward_list, module_dict, attr_dict):
    sub_module_dict = {}
    sub_attr_dict = {}
    for op in forward_list:
        if not "=self." in op:
            continue
        keys = op.split("=self.")
        for k in keys:
            if "(" in k:
                k = k.split("(")[0]
                sub_module_dict[k] = module_dict[k]
            elif "," in k:
                k = k.split(",")[0]
                sub_attr_dict[k] = attr_dict[k]
            elif ")" in k:
                k = k.split(")")[0]
                sub_attr_dict[k] = attr_dict[k]
    return sub_module_dict, sub_attr_dict

# get inputs and outputs of submodule
def get_inputs_outputs(forward_list):
    outputs_dict = {}
    inputs = []
    outputs = []
    for op in forward_list:
        output = op.split("=")[0].strip()
        items = op.replace(output+"=", " ").replace("(", " ").replace(")", " ").replace("[", " ").replace("]", " ").replace(",", " ").split(" ")
        outputs_dict[output] = 0
        for item in items:
            if item.strip() == "":
                continue
            if "." in item:
                item = item.split(".")[0].strip()
            if "=" in item:
                item = item.split("=")[-1].strip()
            if not item.startswith("x"):
                continue
            if item in outputs_dict:
                outputs_dict[item] += 1
            else:
                inputs.append(item)
    for key in outputs_dict:
        if outputs_dict[key] == 0:
            outputs.append(key)
    # print(inputs, outputs)
    return inputs, outputs

# generate file
def generate_file(filename, inputs, outputs, shapes_dict, module_dict, attr_dict, forward_list):
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
            opstr = "        "+op+"\n"
            f.write(opstr.replace(",)\n", ")\n"))
        retstr = "        return"
        for o in outputs:
            if not "return " in retstr:
                retstr = retstr + " " + o
            else:
                retstr = retstr + ", " + o
        f.write(retstr + "\n")
        f.write("\n")
        f.write("m = M().eval()\n")
        for input in inputs:
            in_shape = shapes_dict[input].strip()
            instr = ""
            if "torch.Size" in in_shape:
                if in_shape.startswith("("):
                    instr += "("
                    tuple_items = in_shape.replace("(", "", 1).replace("]), ", "]),  ").split(",  ")
                    for tuple_item in tuple_items:
                        tuple_item = tuple_item.strip()
                        if "torch.Size" in tuple_item:
                            instr += "torch.randn("+tuple_item.strip() + "), "
                        elif tuple_item != "":
                            instr += tuple_item
                else:
                    instr += "torch.randn("+in_shape+")"
            else:
                instr += in_shape
            f.write(input+" = "+instr+"\n")
        f.write("start = time.time()\n")
        f.write("output = m("+inputstr+")\n")
        f.write("end = time.time()\n")
        f.write("print(end-start)\n")

def get_oplists_str(oplists):
    ret=""
    for op in oplists:
        if ret == "":
            ret = ret+op
        else:
            ret = ret + "_" + op
    return ret

def simplify_forward_list(forward_list):
    new_forward_list = []
    replace_dict = {}
    for i in range(len(forward_list)):
        op = forward_list[i]
        if not "(" in op:
            k = op.split("=")[0].strip()
            v = op.split("=")[-1].strip()
            replace_dict[k] = v
        else:
            new_forward_list.append(op)
    for k in replace_dict:
        for j in range(len(new_forward_list)):
            new_forward_list[j] = new_forward_list[j].replace(k+",", replace_dict[k]+",")
    return new_forward_list