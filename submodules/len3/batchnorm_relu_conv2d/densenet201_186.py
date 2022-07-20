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
        self.batchnorm2d187 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu187 = ReLU(inplace=True)
        self.conv2d187 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x660):
        x661=self.batchnorm2d187(x660)
        x662=self.relu187(x661)
        x663=self.conv2d187(x662)
        return x663

m = M().eval()
x660 = torch.randn(torch.Size([1, 128, 7, 7]))
start = time.time()
output = m(x660)
end = time.time()
print(end-start)
