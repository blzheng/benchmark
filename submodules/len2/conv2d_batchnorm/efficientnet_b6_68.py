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
        self.conv2d114 = Conv2d(864, 864, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=864, bias=False)
        self.batchnorm2d68 = BatchNorm2d(864, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x357):
        x358=self.conv2d114(x357)
        x359=self.batchnorm2d68(x358)
        return x359

m = M().eval()
x357 = torch.randn(torch.Size([1, 864, 14, 14]))
start = time.time()
output = m(x357)
end = time.time()
print(end-start)
