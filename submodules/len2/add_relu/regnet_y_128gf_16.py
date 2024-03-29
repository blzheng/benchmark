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
        self.relu68 = ReLU(inplace=True)

    def forward(self, x265, x279):
        x280=operator.add(x265, x279)
        x281=self.relu68(x280)
        return x281

m = M().eval()
x265 = torch.randn(torch.Size([1, 2904, 14, 14]))
x279 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x265, x279)
end = time.time()
print(end-start)
