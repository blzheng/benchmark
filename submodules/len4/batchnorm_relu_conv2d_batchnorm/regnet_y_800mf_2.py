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
        self.batchnorm2d6 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu5 = ReLU(inplace=True)
        self.conv2d9 = Conv2d(144, 144, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=9, bias=False)
        self.batchnorm2d7 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x24):
        x25=self.batchnorm2d6(x24)
        x26=self.relu5(x25)
        x27=self.conv2d9(x26)
        x28=self.batchnorm2d7(x27)
        return x28

m = M().eval()
x24 = torch.randn(torch.Size([1, 144, 56, 56]))
start = time.time()
output = m(x24)
end = time.time()
print(end-start)
