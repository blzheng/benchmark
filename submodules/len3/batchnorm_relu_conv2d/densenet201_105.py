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
        self.batchnorm2d106 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu106 = ReLU(inplace=True)
        self.conv2d106 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x375):
        x376=self.batchnorm2d106(x375)
        x377=self.relu106(x376)
        x378=self.conv2d106(x377)
        return x378

m = M().eval()
x375 = torch.randn(torch.Size([1, 128, 14, 14]))
start = time.time()
output = m(x375)
end = time.time()
print(end-start)
