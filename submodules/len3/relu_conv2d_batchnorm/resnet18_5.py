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
        self.conv2d8 = Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d8 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x26):
        x27=self.relu5(x26)
        x28=self.conv2d8(x27)
        x29=self.batchnorm2d8(x28)
        return x29

m = M().eval()
x26 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x26)
end = time.time()
print(end-start)
