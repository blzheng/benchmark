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
        self.relu52 = ReLU(inplace=True)
        self.conv2d56 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d56 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d57 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x184):
        x185=self.relu52(x184)
        x186=self.conv2d56(x185)
        x187=self.batchnorm2d56(x186)
        x188=self.relu52(x187)
        x189=self.conv2d57(x188)
        return x189

m = M().eval()
x184 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x184)
end = time.time()
print(end-start)
