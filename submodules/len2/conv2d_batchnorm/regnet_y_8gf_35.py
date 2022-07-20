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
        self.conv2d55 = Conv2d(896, 896, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
        self.batchnorm2d35 = BatchNorm2d(896, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x172):
        x173=self.conv2d55(x172)
        x174=self.batchnorm2d35(x173)
        return x174

m = M().eval()
x172 = torch.randn(torch.Size([1, 896, 14, 14]))
start = time.time()
output = m(x172)
end = time.time()
print(end-start)
