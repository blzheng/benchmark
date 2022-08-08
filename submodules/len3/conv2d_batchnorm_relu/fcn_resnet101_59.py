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
        self.conv2d91 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d91 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu88 = ReLU(inplace=True)

    def forward(self, x302):
        x303=self.conv2d91(x302)
        x304=self.batchnorm2d91(x303)
        x305=self.relu88(x304)
        return x305

m = M().eval()
x302 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x302)
end = time.time()
print(end-start)
