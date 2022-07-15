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
        self.conv2d81 = Conv2d(576, 144, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x256):
        x257=self.conv2d81(x256)
        return x257

m = M().eval()
x256 = torch.randn(torch.Size([1, 576, 1, 1]))
start = time.time()
output = m(x256)
end = time.time()
print(end-start)
