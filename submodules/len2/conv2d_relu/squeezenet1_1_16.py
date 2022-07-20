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
        self.conv2d16 = Conv2d(384, 48, kernel_size=(1, 1), stride=(1, 1))
        self.relu16 = ReLU(inplace=True)

    def forward(self, x40):
        x41=self.conv2d16(x40)
        x42=self.relu16(x41)
        return x42

m = M().eval()
x40 = torch.randn(torch.Size([1, 384, 13, 13]))
start = time.time()
output = m(x40)
end = time.time()
print(end-start)