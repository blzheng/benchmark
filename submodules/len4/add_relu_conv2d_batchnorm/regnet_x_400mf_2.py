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
        self.relu9 = ReLU(inplace=True)
        self.conv2d12 = Conv2d(64, 160, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d12 = BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x27, x35):
        x36=operator.add(x27, x35)
        x37=self.relu9(x36)
        x38=self.conv2d12(x37)
        x39=self.batchnorm2d12(x38)
        return x39

m = M().eval()
x27 = torch.randn(torch.Size([1, 64, 28, 28]))
x35 = torch.randn(torch.Size([1, 64, 28, 28]))
start = time.time()
output = m(x27, x35)
end = time.time()
print(end-start)
