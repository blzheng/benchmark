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
        self.linear0 = Linear(in_features=1512, out_features=1000, bias=True)

    def forward(self, x349):
        x350=self.linear0(x349)
        return x350

m = M().eval()
x349 = torch.randn(torch.Size([1, 1512]))
start = time.time()
output = m(x349)
end = time.time()
print(end-start)
