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
        self.relu24 = ReLU(inplace=True)
        self.conv2d27 = Conv2d(192, 432, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d27 = BatchNorm2d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x86):
        x87=self.relu24(x86)
        x88=self.conv2d27(x87)
        x89=self.batchnorm2d27(x88)
        return x89

m = M().eval()
x86 = torch.randn(torch.Size([1, 192, 28, 28]))
start = time.time()
output = m(x86)
end = time.time()
print(end-start)
