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
        self.relu13 = ReLU(inplace=True)
        self.conv2d17 = Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=10, bias=False)
        self.batchnorm2d17 = BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x51):
        x52=self.relu13(x51)
        x53=self.conv2d17(x52)
        x54=self.batchnorm2d17(x53)
        return x54

m = M().eval()
x51 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x51)
end = time.time()
print(end-start)
