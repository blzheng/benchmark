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
        self.sigmoid57 = Sigmoid()
        self.conv2d322 = Conv2d(3840, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x1031, x1027):
        x1032=self.sigmoid57(x1031)
        x1033=operator.mul(x1032, x1027)
        x1034=self.conv2d322(x1033)
        return x1034

m = M().eval()
x1031 = torch.randn(torch.Size([1, 3840, 1, 1]))
x1027 = torch.randn(torch.Size([1, 3840, 7, 7]))
start = time.time()
output = m(x1031, x1027)
end = time.time()
print(end-start)
