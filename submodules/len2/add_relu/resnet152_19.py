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
        self.relu58 = ReLU(inplace=True)

    def forward(self, x208, x200):
        x209=operator.add(x208, x200)
        x210=self.relu58(x209)
        return x210

m = M().eval()
x208 = torch.randn(torch.Size([1, 1024, 14, 14]))
x200 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x208, x200)
end = time.time()
print(end-start)
