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
        self.conv2d99 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4), bias=False)
        self.batchnorm2d99 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu94 = ReLU(inplace=True)
        self.conv2d100 = Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x327):
        x328=self.conv2d99(x327)
        x329=self.batchnorm2d99(x328)
        x330=self.relu94(x329)
        x331=self.conv2d100(x330)
        return x331

m = M().eval()
x327 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x327)
end = time.time()
print(end-start)
