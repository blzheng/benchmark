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
        self.batchnorm2d40 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu49 = ReLU(inplace=True)
        self.conv2d65 = Conv2d(2904, 2904, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=11, bias=False)
        self.batchnorm2d41 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x202):
        x203=self.batchnorm2d40(x202)
        x204=self.relu49(x203)
        x205=self.conv2d65(x204)
        x206=self.batchnorm2d41(x205)
        return x206

m = M().eval()
x202 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x202)
end = time.time()
print(end-start)
