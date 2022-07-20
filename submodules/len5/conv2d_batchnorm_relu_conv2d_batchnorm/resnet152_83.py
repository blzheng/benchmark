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
        self.conv2d128 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d128 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu124 = ReLU(inplace=True)
        self.conv2d129 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d129 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x423):
        x424=self.conv2d128(x423)
        x425=self.batchnorm2d128(x424)
        x426=self.relu124(x425)
        x427=self.conv2d129(x426)
        x428=self.batchnorm2d129(x427)
        return x428

m = M().eval()
x423 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x423)
end = time.time()
print(end-start)
