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
        self.relu5 = ReLU(inplace=True)
        self.conv2d8 = Conv2d(528, 528, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2, bias=False)
        self.batchnorm2d6 = BatchNorm2d(528, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x23):
        x24=self.relu5(x23)
        x25=self.conv2d8(x24)
        x26=self.batchnorm2d6(x25)
        return x26

m = M().eval()
x23 = torch.randn(torch.Size([1, 528, 56, 56]))
start = time.time()
output = m(x23)
end = time.time()
print(end-start)
