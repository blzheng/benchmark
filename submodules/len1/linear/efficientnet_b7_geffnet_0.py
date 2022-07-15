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
        self.linear0 = Linear(in_features=2560, out_features=1000, bias=True)

    def forward(self, x818):
        x819=self.linear0(x818)
        return x819

m = M().eval()
x818 = torch.randn(torch.Size([1, 2560]))
start = time.time()
output = m(x818)
end = time.time()
print(end-start)
