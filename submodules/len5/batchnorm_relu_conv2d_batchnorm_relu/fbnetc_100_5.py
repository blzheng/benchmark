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
        self.batchnorm2d13 = BatchNorm2d(144, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu9 = ReLU(inplace=True)
        self.conv2d14 = Conv2d(144, 144, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=144, bias=False)
        self.batchnorm2d14 = BatchNorm2d(144, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu10 = ReLU(inplace=True)

    def forward(self, x43):
        x44=self.batchnorm2d13(x43)
        x45=self.relu9(x44)
        x46=self.conv2d14(x45)
        x47=self.batchnorm2d14(x46)
        x48=self.relu10(x47)
        return x48

m = M().eval()
x43 = torch.randn(torch.Size([1, 144, 56, 56]))
start = time.time()
output = m(x43)
end = time.time()
print(end-start)
