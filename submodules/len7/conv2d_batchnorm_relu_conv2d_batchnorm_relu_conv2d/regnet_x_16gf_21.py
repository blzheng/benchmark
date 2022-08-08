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
        self.conv2d68 = Conv2d(896, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d68 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu64 = ReLU(inplace=True)
        self.conv2d69 = Conv2d(2048, 2048, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=16, bias=False)
        self.batchnorm2d69 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu65 = ReLU(inplace=True)
        self.conv2d70 = Conv2d(2048, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x219):
        x222=self.conv2d68(x219)
        x223=self.batchnorm2d68(x222)
        x224=self.relu64(x223)
        x225=self.conv2d69(x224)
        x226=self.batchnorm2d69(x225)
        x227=self.relu65(x226)
        x228=self.conv2d70(x227)
        return x228

m = M().eval()
x219 = torch.randn(torch.Size([1, 896, 14, 14]))
start = time.time()
output = m(x219)
end = time.time()
print(end-start)