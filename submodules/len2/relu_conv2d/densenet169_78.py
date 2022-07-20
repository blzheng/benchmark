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
        self.relu79 = ReLU(inplace=True)
        self.conv2d79 = Conv2d(896, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x282):
        x283=self.relu79(x282)
        x284=self.conv2d79(x283)
        return x284

m = M().eval()
x282 = torch.randn(torch.Size([1, 896, 14, 14]))
start = time.time()
output = m(x282)
end = time.time()
print(end-start)
