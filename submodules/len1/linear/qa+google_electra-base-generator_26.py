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
        self.linear26 = Linear(in_features=256, out_features=256, bias=True)

    def forward(self, x197):
        x199=self.linear26(x197)
        return x199

m = M().eval()
x197 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x197)
end = time.time()
print(end-start)
