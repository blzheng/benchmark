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
        self.conv2d128 = Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
        self.batchnorm2d76 = BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x384):
        x385=self.conv2d128(x384)
        x386=self.batchnorm2d76(x385)
        return x386

m = M().eval()
x384 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x384)
end = time.time()
print(end-start)
