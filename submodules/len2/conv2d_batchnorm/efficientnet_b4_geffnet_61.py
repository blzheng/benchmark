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
        self.conv2d103 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d61 = BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x306):
        x307=self.conv2d103(x306)
        x308=self.batchnorm2d61(x307)
        return x308

m = M().eval()
x306 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x306)
end = time.time()
print(end-start)
