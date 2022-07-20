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
        self.relu4 = ReLU(inplace=True)
        self.conv2d7 = Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=4, bias=False)
        self.batchnorm2d7 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu5 = ReLU(inplace=True)

    def forward(self, x19):
        x20=self.relu4(x19)
        x21=self.conv2d7(x20)
        x22=self.batchnorm2d7(x21)
        x23=self.relu5(x22)
        return x23

m = M().eval()
x19 = torch.randn(torch.Size([1, 64, 56, 56]))
start = time.time()
output = m(x19)
end = time.time()
print(end-start)
