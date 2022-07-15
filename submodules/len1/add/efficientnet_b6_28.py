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

    def forward(self, x541, x526):
        x542=operator.add(x541, x526)
        return x542

m = M().eval()
x541 = torch.randn(torch.Size([1, 344, 7, 7]))
x526 = torch.randn(torch.Size([1, 344, 7, 7]))
start = time.time()
output = m(x541, x526)
end = time.time()
print(end-start)
