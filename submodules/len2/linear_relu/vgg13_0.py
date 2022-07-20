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
        self.relu10 = ReLU(inplace=True)

    def forward(self, x27):
        x28=self.linear0(x27)
        x29=self.relu10(x28)
        return x29

m = M().eval()
x27 = torch.randn(torch.Size([1, 25088]))
start = time.time()
output = m(x27)
end = time.time()
print(end-start)
