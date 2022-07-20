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
        self.conv2d110 = Conv2d(336, 336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=14, bias=False)
        self.batchnorm2d68 = BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu86 = ReLU(inplace=True)

    def forward(self, x348):
        x349=self.conv2d110(x348)
        x350=self.batchnorm2d68(x349)
        x351=self.relu86(x350)
        return x351

m = M().eval()
x348 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x348)
end = time.time()
print(end-start)
