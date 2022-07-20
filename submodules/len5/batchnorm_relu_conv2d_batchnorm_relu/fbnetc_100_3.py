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
        self.batchnorm2d7 = BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu5 = ReLU(inplace=True)
        self.conv2d8 = Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False)
        self.batchnorm2d8 = BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu6 = ReLU(inplace=True)

    def forward(self, x23):
        x24=self.batchnorm2d7(x23)
        x25=self.relu5(x24)
        x26=self.conv2d8(x25)
        x27=self.batchnorm2d8(x26)
        x28=self.relu6(x27)
        return x28

m = M().eval()
x23 = torch.randn(torch.Size([1, 24, 56, 56]))
start = time.time()
output = m(x23)
end = time.time()
print(end-start)
