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

    def forward(self, x220):
        x221=torch.nn.functional.relu(x220,inplace=True)
        return x221

m = M().eval()
x220 = torch.randn(torch.Size([1, 192, 12, 12]))
start = time.time()
output = m(x220)
end = time.time()
print(end-start)
