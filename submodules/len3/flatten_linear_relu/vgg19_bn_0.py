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
        self.linear0 = Linear(in_features=25088, out_features=4096, bias=True)
        self.relu16 = ReLU(inplace=True)

    def forward(self, x54):
        x55=torch.flatten(x54, 1)
        x56=self.linear0(x55)
        x57=self.relu16(x56)
        return x57

m = M().eval()
x54 = torch.randn(torch.Size([1, 512, 7, 7]))
start = time.time()
output = m(x54)
end = time.time()
print(end-start)
