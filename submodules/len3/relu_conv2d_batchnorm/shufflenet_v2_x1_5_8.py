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
        self.relu19 = ReLU(inplace=True)
        self.conv2d30 = Conv2d(176, 176, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=176, bias=False)
        self.batchnorm2d30 = BatchNorm2d(176, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x189):
        x190=self.relu19(x189)
        x191=self.conv2d30(x190)
        x192=self.batchnorm2d30(x191)
        return x192

m = M().eval()
x189 = torch.randn(torch.Size([1, 176, 14, 14]))
start = time.time()
output = m(x189)
end = time.time()
print(end-start)
