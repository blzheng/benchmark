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

    def forward(self, x43, x45):
        x46=torch.cat([x43, x45], 1)
        return x46

m = M().eval()
x43 = torch.randn(torch.Size([1, 192, 27, 27]))
x45 = torch.randn(torch.Size([1, 192, 27, 27]))
start = time.time()
output = m(x43, x45)
end = time.time()
print(end-start)
