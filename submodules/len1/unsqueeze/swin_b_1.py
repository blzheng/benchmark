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

    def forward(self, x33):
        x34=x33.unsqueeze(0)
        return x34

m = M().eval()
x33 = torch.randn(torch.Size([4, 49, 49]))
start = time.time()
output = m(x33)
end = time.time()
print(end-start)
