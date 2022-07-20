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
        self.conv2d22 = Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
        self.batchnorm2d22 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu615 = ReLU6(inplace=True)

    def forward(self, x62):
        x63=self.conv2d22(x62)
        x64=self.batchnorm2d22(x63)
        x65=self.relu615(x64)
        return x65

m = M().eval()
x62 = torch.randn(torch.Size([1, 384, 14, 14]))
start = time.time()
output = m(x62)
end = time.time()
print(end-start)
