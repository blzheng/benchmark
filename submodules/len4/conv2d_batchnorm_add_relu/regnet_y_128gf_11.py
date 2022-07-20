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
        self.conv2d48 = Conv2d(1056, 2904, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d30 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu40 = ReLU(inplace=True)

    def forward(self, x151, x167):
        x152=self.conv2d48(x151)
        x153=self.batchnorm2d30(x152)
        x168=operator.add(x153, x167)
        x169=self.relu40(x168)
        return x169

m = M().eval()
x151 = torch.randn(torch.Size([1, 1056, 28, 28]))
x167 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x151, x167)
end = time.time()
print(end-start)
