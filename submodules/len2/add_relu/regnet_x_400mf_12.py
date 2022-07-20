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
        self.relu39 = ReLU(inplace=True)

    def forward(self, x131, x139):
        x140=operator.add(x131, x139)
        x141=self.relu39(x140)
        return x141

m = M().eval()
x131 = torch.randn(torch.Size([1, 400, 7, 7]))
x139 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x131, x139)
end = time.time()
print(end-start)
