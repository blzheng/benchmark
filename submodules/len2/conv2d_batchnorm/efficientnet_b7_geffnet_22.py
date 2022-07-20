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
        self.conv2d38 = Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=288, bias=False)
        self.batchnorm2d22 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x116):
        x117=self.conv2d38(x116)
        x118=self.batchnorm2d22(x117)
        return x118

m = M().eval()
x116 = torch.randn(torch.Size([1, 288, 56, 56]))
start = time.time()
output = m(x116)
end = time.time()
print(end-start)
