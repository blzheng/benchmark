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
        self.batchnorm2d185 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu185 = ReLU(inplace=True)
        self.conv2d185 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x653):
        x654=self.batchnorm2d185(x653)
        x655=self.relu185(x654)
        x656=self.conv2d185(x655)
        return x656

m = M().eval()
x653 = torch.randn(torch.Size([1, 128, 7, 7]))
start = time.time()
output = m(x653)
end = time.time()
print(end-start)
