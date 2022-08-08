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
        self.relu16 = ReLU(inplace=True)
        self.conv2d20 = Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d20 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x65):
        x66=self.relu16(x65)
        x67=self.conv2d20(x66)
        x68=self.batchnorm2d20(x67)
        return x68

m = M().eval()
x65 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x65)
end = time.time()
print(end-start)
