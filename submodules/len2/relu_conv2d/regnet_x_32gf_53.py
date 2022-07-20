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
        self.relu53 = ReLU(inplace=True)
        self.conv2d57 = Conv2d(1344, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x184):
        x185=self.relu53(x184)
        x186=self.conv2d57(x185)
        return x186

m = M().eval()
x184 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x184)
end = time.time()
print(end-start)
