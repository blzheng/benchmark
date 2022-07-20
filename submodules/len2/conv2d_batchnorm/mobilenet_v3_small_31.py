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
        self.conv2d47 = Conv2d(576, 576, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=576, bias=False)
        self.batchnorm2d31 = BatchNorm2d(576, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x136):
        x137=self.conv2d47(x136)
        x138=self.batchnorm2d31(x137)
        return x138

m = M().eval()
x136 = torch.randn(torch.Size([1, 576, 7, 7]))
start = time.time()
output = m(x136)
end = time.time()
print(end-start)
