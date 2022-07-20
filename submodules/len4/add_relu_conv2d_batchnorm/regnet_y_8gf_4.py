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
        self.relu20 = ReLU(inplace=True)
        self.conv2d28 = Conv2d(448, 448, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d18 = BatchNorm2d(448, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x71, x85):
        x86=operator.add(x71, x85)
        x87=self.relu20(x86)
        x88=self.conv2d28(x87)
        x89=self.batchnorm2d18(x88)
        return x89

m = M().eval()
x71 = torch.randn(torch.Size([1, 448, 28, 28]))
x85 = torch.randn(torch.Size([1, 448, 28, 28]))
start = time.time()
output = m(x71, x85)
end = time.time()
print(end-start)
