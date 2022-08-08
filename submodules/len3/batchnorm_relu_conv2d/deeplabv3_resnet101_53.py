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
        self.batchnorm2d83 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu79 = ReLU(inplace=True)
        self.conv2d84 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x276):
        x277=self.batchnorm2d83(x276)
        x278=self.relu79(x277)
        x279=self.conv2d84(x278)
        return x279

m = M().eval()
x276 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x276)
end = time.time()
print(end-start)
