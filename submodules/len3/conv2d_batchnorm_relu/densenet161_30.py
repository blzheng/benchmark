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
        self.conv2d61 = Conv2d(912, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d62 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu62 = ReLU(inplace=True)

    def forward(self, x220):
        x221=self.conv2d61(x220)
        x222=self.batchnorm2d62(x221)
        x223=self.relu62(x222)
        return x223

m = M().eval()
x220 = torch.randn(torch.Size([1, 912, 14, 14]))
start = time.time()
output = m(x220)
end = time.time()
print(end-start)
