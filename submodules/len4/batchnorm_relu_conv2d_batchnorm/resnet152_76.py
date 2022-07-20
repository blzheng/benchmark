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
        self.batchnorm2d118 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu115 = ReLU(inplace=True)
        self.conv2d119 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d119 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x391):
        x392=self.batchnorm2d118(x391)
        x393=self.relu115(x392)
        x394=self.conv2d119(x393)
        x395=self.batchnorm2d119(x394)
        return x395

m = M().eval()
x391 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x391)
end = time.time()
print(end-start)
