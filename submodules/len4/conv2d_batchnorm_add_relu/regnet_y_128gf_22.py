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
        self.conv2d103 = Conv2d(2904, 2904, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d63 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu80 = ReLU(inplace=True)

    def forward(self, x325, x313):
        x326=self.conv2d103(x325)
        x327=self.batchnorm2d63(x326)
        x328=operator.add(x313, x327)
        x329=self.relu80(x328)
        return x329

m = M().eval()
x325 = torch.randn(torch.Size([1, 2904, 14, 14]))
x313 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x325, x313)
end = time.time()
print(end-start)
