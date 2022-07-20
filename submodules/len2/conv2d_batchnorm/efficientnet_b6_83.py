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
        self.conv2d139 = Conv2d(1200, 1200, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1200, bias=False)
        self.batchnorm2d83 = BatchNorm2d(1200, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x435):
        x436=self.conv2d139(x435)
        x437=self.batchnorm2d83(x436)
        return x437

m = M().eval()
x435 = torch.randn(torch.Size([1, 1200, 14, 14]))
start = time.time()
output = m(x435)
end = time.time()
print(end-start)
