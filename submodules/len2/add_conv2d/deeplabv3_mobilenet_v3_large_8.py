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
        self.conv2d56 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x164, x150):
        x165=operator.add(x164, x150)
        x166=self.conv2d56(x165)
        return x166

m = M().eval()
x164 = torch.randn(torch.Size([1, 160, 14, 14]))
x150 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x164, x150)
end = time.time()
print(end-start)
