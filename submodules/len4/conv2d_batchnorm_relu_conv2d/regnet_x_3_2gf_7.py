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
        self.conv2d12 = Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d12 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu10 = ReLU(inplace=True)
        self.conv2d13 = Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)

    def forward(self, x37):
        x38=self.conv2d12(x37)
        x39=self.batchnorm2d12(x38)
        x40=self.relu10(x39)
        x41=self.conv2d13(x40)
        return x41

m = M().eval()
x37 = torch.randn(torch.Size([1, 192, 28, 28]))
start = time.time()
output = m(x37)
end = time.time()
print(end-start)
