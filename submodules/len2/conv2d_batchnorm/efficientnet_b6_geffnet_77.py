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
        self.conv2d129 = Conv2d(1200, 1200, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1200, bias=False)
        self.batchnorm2d77 = BatchNorm2d(1200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x385):
        x386=self.conv2d129(x385)
        x387=self.batchnorm2d77(x386)
        return x387

m = M().eval()
x385 = torch.randn(torch.Size([1, 1200, 14, 14]))
start = time.time()
output = m(x385)
end = time.time()
print(end-start)
