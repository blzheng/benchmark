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
        self.conv2d209 = Conv2d(2064, 2064, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2064, bias=False)
        self.batchnorm2d125 = BatchNorm2d(2064, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x624):
        x625=self.conv2d209(x624)
        x626=self.batchnorm2d125(x625)
        return x626

m = M().eval()
x624 = torch.randn(torch.Size([1, 2064, 7, 7]))
start = time.time()
output = m(x624)
end = time.time()
print(end-start)
