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
        self.linear0 = Linear(in_features=1664, out_features=1000, bias=True)

    def forward(self, x597):
        x598=torch.flatten(x597, 1)
        x599=self.linear0(x598)
        return x599

m = M().eval()
x597 = torch.randn(torch.Size([1, 1664, 1, 1]))
start = time.time()
output = m(x597)
end = time.time()
print(end-start)
