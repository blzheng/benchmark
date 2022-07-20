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
        self.linear0 = Linear(in_features=9216, out_features=4096, bias=True)
        self.relu5 = ReLU(inplace=True)

    def forward(self, x16):
        x17=self.linear0(x16)
        x18=self.relu5(x17)
        return x18

m = M().eval()
x16 = torch.randn(torch.Size([1, 9216]))
start = time.time()
output = m(x16)
end = time.time()
print(end-start)
