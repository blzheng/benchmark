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
        self.relu24 = ReLU(inplace=True)
        self.conv2d37 = Conv2d(288, 288, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=288, bias=False)
        self.batchnorm2d37 = BatchNorm2d(288, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu25 = ReLU(inplace=True)

    def forward(self, x105):
        x106=self.relu24(x105)
        x107=self.conv2d37(x106)
        x108=self.batchnorm2d37(x107)
        x109=self.relu25(x108)
        return x109

m = M().eval()
x105 = torch.randn(torch.Size([1, 288, 14, 14]))
start = time.time()
output = m(x105)
end = time.time()
print(end-start)
