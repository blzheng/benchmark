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
        self.conv2d65 = Conv2d(720, 720, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=720, bias=False)
        self.batchnorm2d39 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x192):
        x193=self.conv2d65(x192)
        x194=self.batchnorm2d39(x193)
        return x194

m = M().eval()
x192 = torch.randn(torch.Size([1, 720, 14, 14]))
start = time.time()
output = m(x192)
end = time.time()
print(end-start)
