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
        self.conv2d195 = Conv2d(1824, 1824, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1824, bias=False)
        self.batchnorm2d127 = BatchNorm2d(1824, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x625):
        x626=self.conv2d195(x625)
        x627=self.batchnorm2d127(x626)
        return x627

m = M().eval()
x625 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x625)
end = time.time()
print(end-start)
