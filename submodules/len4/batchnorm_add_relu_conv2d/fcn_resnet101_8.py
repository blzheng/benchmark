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
        self.batchnorm2d23 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu19 = ReLU(inplace=True)
        self.conv2d24 = Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x77, x70):
        x78=self.batchnorm2d23(x77)
        x79=operator.add(x78, x70)
        x80=self.relu19(x79)
        x81=self.conv2d24(x80)
        return x81

m = M().eval()
x77 = torch.randn(torch.Size([1, 512, 28, 28]))
x70 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x77, x70)
end = time.time()
print(end-start)
