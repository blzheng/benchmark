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
        self.conv2d6 = Conv2d(72, 72, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=3, bias=False)
        self.batchnorm2d6 = BatchNorm2d(72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu5 = ReLU(inplace=True)

    def forward(self, x18):
        x19=self.conv2d6(x18)
        x20=self.batchnorm2d6(x19)
        x21=self.relu5(x20)
        return x21

m = M().eval()
x18 = torch.randn(torch.Size([1, 72, 56, 56]))
start = time.time()
output = m(x18)
end = time.time()
print(end-start)
