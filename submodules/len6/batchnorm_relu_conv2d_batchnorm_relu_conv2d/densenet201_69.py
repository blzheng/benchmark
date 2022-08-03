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
        self.batchnorm2d142 = BatchNorm2d(992, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu142 = ReLU(inplace=True)
        self.conv2d142 = Conv2d(992, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d143 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu143 = ReLU(inplace=True)
        self.conv2d143 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x503):
        x504=self.batchnorm2d142(x503)
        x505=self.relu142(x504)
        x506=self.conv2d142(x505)
        x507=self.batchnorm2d143(x506)
        x508=self.relu143(x507)
        x509=self.conv2d143(x508)
        return x509

m = M().eval()
x503 = torch.randn(torch.Size([1, 992, 7, 7]))
start = time.time()
output = m(x503)
end = time.time()
print(end-start)
