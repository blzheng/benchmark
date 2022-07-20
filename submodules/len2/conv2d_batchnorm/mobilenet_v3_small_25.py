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
        self.conv2d37 = Conv2d(288, 288, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=288, bias=False)
        self.batchnorm2d25 = BatchNorm2d(288, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x107):
        x108=self.conv2d37(x107)
        x109=self.batchnorm2d25(x108)
        return x109

m = M().eval()
x107 = torch.randn(torch.Size([1, 288, 14, 14]))
start = time.time()
output = m(x107)
end = time.time()
print(end-start)
