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
        self.batchnorm2d6 = BatchNorm2d(104, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu5 = ReLU(inplace=True)
        self.conv2d9 = Conv2d(104, 104, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=13, bias=False)

    def forward(self, x24):
        x25=self.batchnorm2d6(x24)
        x26=self.relu5(x25)
        x27=self.conv2d9(x26)
        return x27

m = M().eval()
x24 = torch.randn(torch.Size([1, 104, 56, 56]))
start = time.time()
output = m(x24)
end = time.time()
print(end-start)
