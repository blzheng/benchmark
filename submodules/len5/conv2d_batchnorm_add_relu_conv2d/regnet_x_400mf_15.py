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
        self.conv2d40 = Conv2d(400, 400, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d40 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu36 = ReLU(inplace=True)
        self.conv2d41 = Conv2d(400, 400, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x127, x121):
        x128=self.conv2d40(x127)
        x129=self.batchnorm2d40(x128)
        x130=operator.add(x121, x129)
        x131=self.relu36(x130)
        x132=self.conv2d41(x131)
        return x132

m = M().eval()
x127 = torch.randn(torch.Size([1, 400, 7, 7]))
x121 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x127, x121)
end = time.time()
print(end-start)
