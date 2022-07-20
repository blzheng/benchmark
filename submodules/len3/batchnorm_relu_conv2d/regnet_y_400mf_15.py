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
        self.batchnorm2d47 = BatchNorm2d(440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu57 = ReLU(inplace=True)
        self.conv2d76 = Conv2d(440, 440, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=55, bias=False)

    def forward(self, x236):
        x237=self.batchnorm2d47(x236)
        x238=self.relu57(x237)
        x239=self.conv2d76(x238)
        return x239

m = M().eval()
x236 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x236)
end = time.time()
print(end-start)
