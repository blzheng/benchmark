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
        self.linear3 = Linear(in_features=768, out_features=192, bias=True)

    def forward(self, x23):
        x24=self.linear3(x23)
        x25=torch.permute(x24, [0, 3, 1, 2])
        return x25

m = M().eval()
x23 = torch.randn(torch.Size([1, 56, 56, 768]))
start = time.time()
output = m(x23)
end = time.time()
print(end-start)
