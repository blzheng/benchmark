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
        self.linear1 = Linear(in_features=4096, out_features=4096, bias=True)

    def forward(self, x30):
        x31=self.linear1(x30)
        return x31

m = M().eval()
x30 = torch.randn(torch.Size([1, 4096]))
start = time.time()
output = m(x30)
end = time.time()
print(end-start)
