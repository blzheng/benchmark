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
        self.linear4 = Linear(in_features=256, out_features=256, bias=True)

    def forward(self, x57, x60):
        x61=x57.view(x60)
        x62=self.linear4(x61)
        return x62

m = M().eval()
x57 = torch.randn(torch.Size([1, 384, 4, 64]))
x60 = (1, 384, 256, )
start = time.time()
output = m(x57, x60)
end = time.time()
print(end-start)
