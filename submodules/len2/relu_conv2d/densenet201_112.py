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
        self.relu113 = ReLU(inplace=True)
        self.conv2d113 = Conv2d(1440, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x401):
        x402=self.relu113(x401)
        x403=self.conv2d113(x402)
        return x403

m = M().eval()
x401 = torch.randn(torch.Size([1, 1440, 14, 14]))
start = time.time()
output = m(x401)
end = time.time()
print(end-start)
