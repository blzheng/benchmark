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
        self.conv2d69 = Conv2d(736, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d70 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu70 = ReLU(inplace=True)
        self.conv2d70 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x248):
        x249=self.conv2d69(x248)
        x250=self.batchnorm2d70(x249)
        x251=self.relu70(x250)
        x252=self.conv2d70(x251)
        return x252

m = M().eval()
x248 = torch.randn(torch.Size([1, 736, 14, 14]))
start = time.time()
output = m(x248)
end = time.time()
print(end-start)
