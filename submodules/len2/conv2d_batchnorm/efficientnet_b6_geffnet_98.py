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
        self.conv2d164 = Conv2d(2064, 2064, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2064, bias=False)
        self.batchnorm2d98 = BatchNorm2d(2064, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x489):
        x490=self.conv2d164(x489)
        x491=self.batchnorm2d98(x490)
        return x491

m = M().eval()
x489 = torch.randn(torch.Size([1, 2064, 7, 7]))
start = time.time()
output = m(x489)
end = time.time()
print(end-start)
