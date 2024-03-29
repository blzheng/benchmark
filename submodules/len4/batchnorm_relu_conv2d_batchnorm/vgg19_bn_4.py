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
        self.batchnorm2d6 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu6 = ReLU(inplace=True)
        self.conv2d7 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm2d7 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x21):
        x22=self.batchnorm2d6(x21)
        x23=self.relu6(x22)
        x24=self.conv2d7(x23)
        x25=self.batchnorm2d7(x24)
        return x25

m = M().eval()
x21 = torch.randn(torch.Size([1, 256, 56, 56]))
start = time.time()
output = m(x21)
end = time.time()
print(end-start)
