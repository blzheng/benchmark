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
        self.batchnorm2d30 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu20 = ReLU(inplace=True)
        self.conv2d31 = Conv2d(240, 240, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=240, bias=False)

    def forward(self, x98):
        x99=self.batchnorm2d30(x98)
        x100=self.relu20(x99)
        x101=self.conv2d31(x100)
        return x101

m = M().eval()
x98 = torch.randn(torch.Size([1, 240, 14, 14]))
start = time.time()
output = m(x98)
end = time.time()
print(end-start)
