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
        self.relu64 = ReLU(inplace=True)
        self.conv2d68 = Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        self.batchnorm2d68 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d69 = Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d69 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x222):
        x223=self.relu64(x222)
        x224=self.conv2d68(x223)
        x225=self.batchnorm2d68(x224)
        x226=self.relu64(x225)
        x227=self.conv2d69(x226)
        x228=self.batchnorm2d69(x227)
        return x228

m = M().eval()
x222 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x222)
end = time.time()
print(end-start)
