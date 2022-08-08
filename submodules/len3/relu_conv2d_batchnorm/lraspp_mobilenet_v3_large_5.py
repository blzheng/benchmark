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
        self.conv2d10 = Conv2d(72, 72, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=72, bias=False)
        self.batchnorm2d10 = BatchNorm2d(72, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x28):
        x29=self.relu5(x28)
        x30=self.conv2d10(x29)
        x31=self.batchnorm2d10(x30)
        return x31

m = M().eval()
x28 = torch.randn(torch.Size([1, 72, 56, 56]))
start = time.time()
output = m(x28)
end = time.time()
print(end-start)
