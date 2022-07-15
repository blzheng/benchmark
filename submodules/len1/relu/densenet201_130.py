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
        self.relu130 = ReLU(inplace=True)

    def forward(self, x460):
        x461=self.relu130(x460)
        return x461

m = M().eval()
x460 = torch.randn(torch.Size([1, 128, 14, 14]))
start = time.time()
output = m(x460)
end = time.time()
print(end-start)
