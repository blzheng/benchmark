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
        self.conv2d144 = Conv2d(1344, 1344, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1344, bias=False)
        self.batchnorm2d100 = BatchNorm2d(1344, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x468):
        x469=self.conv2d144(x468)
        x470=self.batchnorm2d100(x469)
        return x470

m = M().eval()
x468 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x468)
end = time.time()
print(end-start)
