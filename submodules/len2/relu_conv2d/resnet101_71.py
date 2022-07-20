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
        self.relu70 = ReLU(inplace=True)
        self.conv2d76 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x249):
        x250=self.relu70(x249)
        x251=self.conv2d76(x250)
        return x251

m = M().eval()
x249 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x249)
end = time.time()
print(end-start)
