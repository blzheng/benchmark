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
        self.conv2d88 = Conv2d(224, 2016, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid16 = Sigmoid()

    def forward(self, x276):
        x277=self.conv2d88(x276)
        x278=self.sigmoid16(x277)
        return x278

m = M().eval()
x276 = torch.randn(torch.Size([1, 224, 1, 1]))
start = time.time()
output = m(x276)
end = time.time()
print(end-start)
