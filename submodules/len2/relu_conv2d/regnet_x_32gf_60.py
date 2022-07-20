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
        self.relu60 = ReLU(inplace=True)
        self.conv2d64 = Conv2d(1344, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x208):
        x209=self.relu60(x208)
        x210=self.conv2d64(x209)
        return x210

m = M().eval()
x208 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x208)
end = time.time()
print(end-start)
