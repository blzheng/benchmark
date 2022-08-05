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
        self.relu97 = ReLU(inplace=True)

    def forward(self, x342, x334):
        x343=operator.add(x342, x334)
        x344=self.relu97(x343)
        return x344

m = M().eval()
x342 = torch.randn(torch.Size([1, 2048, 28, 28]))
x334 = torch.randn(torch.Size([1, 2048, 28, 28]))
start = time.time()
output = m(x342, x334)
end = time.time()
print(end-start)
