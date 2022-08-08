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
        self.conv2d64 = Conv2d(960, 256, kernel_size=(3, 3), stride=(1, 1), padding=(24, 24), dilation=(24, 24), bias=False)
        self.batchnorm2d48 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu21 = ReLU()

    def forward(self, x183):
        x190=self.conv2d64(x183)
        x191=self.batchnorm2d48(x190)
        x192=self.relu21(x191)
        return x192

m = M().eval()
x183 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x183)
end = time.time()
print(end-start)
