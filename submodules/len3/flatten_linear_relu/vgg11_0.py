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
        self.relu8 = ReLU(inplace=True)

    def forward(self, x22):
        x23=torch.flatten(x22, 1)
        x24=self.linear0(x23)
        x25=self.relu8(x24)
        return x25

m = M().eval()
x22 = torch.randn(torch.Size([1, 512, 7, 7]))
start = time.time()
output = m(x22)
end = time.time()
print(end-start)
