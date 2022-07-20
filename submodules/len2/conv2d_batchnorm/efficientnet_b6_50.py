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
        self.conv2d84 = Conv2d(864, 864, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=864, bias=False)
        self.batchnorm2d50 = BatchNorm2d(864, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x261):
        x262=self.conv2d84(x261)
        x263=self.batchnorm2d50(x262)
        return x263

m = M().eval()
x261 = torch.randn(torch.Size([1, 864, 14, 14]))
start = time.time()
output = m(x261)
end = time.time()
print(end-start)
