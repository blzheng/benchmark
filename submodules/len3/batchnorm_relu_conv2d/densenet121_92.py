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
        self.batchnorm2d93 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu93 = ReLU(inplace=True)
        self.conv2d93 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x331):
        x332=self.batchnorm2d93(x331)
        x333=self.relu93(x332)
        x334=self.conv2d93(x333)
        return x334

m = M().eval()
x331 = torch.randn(torch.Size([1, 128, 7, 7]))
start = time.time()
output = m(x331)
end = time.time()
print(end-start)
