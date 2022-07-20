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
        self.linear0 = Linear(in_features=512, out_features=1000, bias=True)

    def forward(self, x67):
        x68=torch.flatten(x67, 1)
        x69=self.linear0(x68)
        return x69

m = M().eval()
x67 = torch.randn(torch.Size([1, 512, 1, 1]))
start = time.time()
output = m(x67)
end = time.time()
print(end-start)
