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
        self.sigmoid52 = Sigmoid()
        self.conv2d261 = Conv2d(3840, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x819, x815):
        x820=self.sigmoid52(x819)
        x821=operator.mul(x820, x815)
        x822=self.conv2d261(x821)
        return x822

m = M().eval()
x819 = torch.randn(torch.Size([1, 3840, 1, 1]))
x815 = torch.randn(torch.Size([1, 3840, 7, 7]))
start = time.time()
output = m(x819, x815)
end = time.time()
print(end-start)
