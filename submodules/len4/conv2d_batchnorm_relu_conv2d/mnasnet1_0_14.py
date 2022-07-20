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
        self.conv2d21 = Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d21 = BatchNorm2d(240, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu14 = ReLU(inplace=True)
        self.conv2d22 = Conv2d(240, 240, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=240, bias=False)

    def forward(self, x60):
        x61=self.conv2d21(x60)
        x62=self.batchnorm2d21(x61)
        x63=self.relu14(x62)
        x64=self.conv2d22(x63)
        return x64

m = M().eval()
x60 = torch.randn(torch.Size([1, 40, 28, 28]))
start = time.time()
output = m(x60)
end = time.time()
print(end-start)
