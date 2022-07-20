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
        self.conv2d22 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        self.batchnorm2d22 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu19 = ReLU(inplace=True)
        self.conv2d23 = Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d23 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x71):
        x72=self.conv2d22(x71)
        x73=self.batchnorm2d22(x72)
        x74=self.relu19(x73)
        x75=self.conv2d23(x74)
        x76=self.batchnorm2d23(x75)
        return x76

m = M().eval()
x71 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x71)
end = time.time()
print(end-start)
