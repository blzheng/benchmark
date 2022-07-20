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
        self.batchnorm2d11 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu11 = ReLU(inplace=True)
        self.conv2d12 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm2d12 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu12 = ReLU(inplace=True)

    def forward(self, x38):
        x39=self.batchnorm2d11(x38)
        x40=self.relu11(x39)
        x41=self.conv2d12(x40)
        x42=self.batchnorm2d12(x41)
        x43=self.relu12(x42)
        return x43

m = M().eval()
x38 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x38)
end = time.time()
print(end-start)
