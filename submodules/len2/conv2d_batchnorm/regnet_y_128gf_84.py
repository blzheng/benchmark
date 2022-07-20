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
        self.conv2d136 = Conv2d(7392, 7392, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=28, bias=False)
        self.batchnorm2d84 = BatchNorm2d(7392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x430):
        x431=self.conv2d136(x430)
        x432=self.batchnorm2d84(x431)
        return x432

m = M().eval()
x430 = torch.randn(torch.Size([1, 7392, 14, 14]))
start = time.time()
output = m(x430)
end = time.time()
print(end-start)
