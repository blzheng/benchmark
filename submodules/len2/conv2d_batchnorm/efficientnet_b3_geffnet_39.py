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
        self.conv2d65 = Conv2d(576, 576, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=576, bias=False)
        self.batchnorm2d39 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x193):
        x194=self.conv2d65(x193)
        x195=self.batchnorm2d39(x194)
        return x195

m = M().eval()
x193 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x193)
end = time.time()
print(end-start)
