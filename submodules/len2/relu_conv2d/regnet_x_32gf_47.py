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
        self.relu47 = ReLU(inplace=True)
        self.conv2d51 = Conv2d(1344, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x164):
        x165=self.relu47(x164)
        x166=self.conv2d51(x165)
        return x166

m = M().eval()
x164 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x164)
end = time.time()
print(end-start)
