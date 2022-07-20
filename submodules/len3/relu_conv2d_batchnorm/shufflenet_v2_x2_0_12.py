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
        self.relu28 = ReLU(inplace=True)
        self.conv2d44 = Conv2d(488, 488, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=488, bias=False)
        self.batchnorm2d44 = BatchNorm2d(488, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x279):
        x280=self.relu28(x279)
        x281=self.conv2d44(x280)
        x282=self.batchnorm2d44(x281)
        return x282

m = M().eval()
x279 = torch.randn(torch.Size([1, 488, 14, 14]))
start = time.time()
output = m(x279)
end = time.time()
print(end-start)
