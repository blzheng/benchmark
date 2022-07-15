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
        self.conv2d1 = Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x3):
        x4=self.conv2d1(x3)
        return x4

m = M().eval()
x3 = torch.randn(torch.Size([1, 64, 224, 224]))
start = time.time()
output = m(x3)
end = time.time()
print(end-start)
