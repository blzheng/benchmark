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
        self.batchnorm2d127 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu127 = ReLU(inplace=True)
        self.conv2d127 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x450):
        x451=self.batchnorm2d127(x450)
        x452=self.relu127(x451)
        x453=self.conv2d127(x452)
        return x453

m = M().eval()
x450 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x450)
end = time.time()
print(end-start)
