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
        self.relu16 = ReLU(inplace=True)
        self.conv2d23 = Conv2d(120, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x70):
        x71=self.relu16(x70)
        x72=self.conv2d23(x71)
        return x72

m = M().eval()
x70 = torch.randn(torch.Size([1, 120, 28, 28]))
start = time.time()
output = m(x70)
end = time.time()
print(end-start)
