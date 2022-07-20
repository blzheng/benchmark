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
        self.conv2d172 = Conv2d(1472, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d173 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu173 = ReLU(inplace=True)
        self.conv2d173 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x610):
        x611=self.conv2d172(x610)
        x612=self.batchnorm2d173(x611)
        x613=self.relu173(x612)
        x614=self.conv2d173(x613)
        return x614

m = M().eval()
x610 = torch.randn(torch.Size([1, 1472, 7, 7]))
start = time.time()
output = m(x610)
end = time.time()
print(end-start)
