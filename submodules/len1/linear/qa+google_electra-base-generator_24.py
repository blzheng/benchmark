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
        self.linear24 = Linear(in_features=1024, out_features=256, bias=True)

    def forward(self, x193):
        x194=self.linear24(x193)
        return x194

m = M().eval()
x193 = torch.randn(torch.Size([1, 384, 1024]))
start = time.time()
output = m(x193)
end = time.time()
print(end-start)
