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
        self.batchnorm2d199 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu199 = ReLU(inplace=True)
        self.conv2d199 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x702):
        x703=self.batchnorm2d199(x702)
        x704=self.relu199(x703)
        x705=self.conv2d199(x704)
        return x705

m = M().eval()
x702 = torch.randn(torch.Size([1, 128, 7, 7]))
start = time.time()
output = m(x702)
end = time.time()
print(end-start)
