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
        self.conv2d111 = Conv2d(336, 84, kernel_size=(1, 1), stride=(1, 1))
        self.relu87 = ReLU()
        self.conv2d112 = Conv2d(84, 336, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid21 = Sigmoid()

    def forward(self, x352, x351):
        x353=self.conv2d111(x352)
        x354=self.relu87(x353)
        x355=self.conv2d112(x354)
        x356=self.sigmoid21(x355)
        x357=operator.mul(x356, x351)
        return x357

m = M().eval()
x352 = torch.randn(torch.Size([1, 336, 1, 1]))
x351 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x352, x351)
end = time.time()
print(end-start)
