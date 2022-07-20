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
        self.relu192 = ReLU(inplace=True)
        self.conv2d192 = Conv2d(1792, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d193 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu193 = ReLU(inplace=True)

    def forward(self, x679):
        x680=self.relu192(x679)
        x681=self.conv2d192(x680)
        x682=self.batchnorm2d193(x681)
        x683=self.relu193(x682)
        return x683

m = M().eval()
x679 = torch.randn(torch.Size([1, 1792, 7, 7]))
start = time.time()
output = m(x679)
end = time.time()
print(end-start)
