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
        self.batchnorm2d47 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu43 = ReLU(inplace=True)
        self.conv2d48 = Conv2d(400, 400, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=25, bias=False)
        self.batchnorm2d48 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x152):
        x153=self.batchnorm2d47(x152)
        x154=self.relu43(x153)
        x155=self.conv2d48(x154)
        x156=self.batchnorm2d48(x155)
        return x156

m = M().eval()
x152 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x152)
end = time.time()
print(end-start)
