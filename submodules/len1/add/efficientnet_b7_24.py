import torch
from torch import tensor
import torch.nn as nn
from torch.nn import *
import torchvision
import torchvision.models as models
from torchvision.ops.stochastic_depth import stochastic_depth
import time
import builtins
import operator

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()

    def forward(self, x460, x445):
        x461=operator.add(x460, x445)
        return x461

m = M().eval()
x460 = torch.randn(torch.Size([1, 224, 14, 14]))
x445 = torch.randn(torch.Size([1, 224, 14, 14]))
start = time.time()
output = m(x460, x445)
end = time.time()
print(end-start)
