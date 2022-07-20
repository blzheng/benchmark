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
        self.relu7 = ReLU(inplace=True)
        self.conv2d10 = Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchnorm2d10 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu9 = ReLU(inplace=True)
        self.conv2d11 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x33):
        x34=self.relu7(x33)
        x35=self.conv2d10(x34)
        x36=self.batchnorm2d10(x35)
        x37=self.relu9(x36)
        x38=self.conv2d11(x37)
        return x38

m = M().eval()
x33 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x33)
end = time.time()
print(end-start)
