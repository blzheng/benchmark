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
        self.batchnorm2d35 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu31 = ReLU(inplace=True)
        self.conv2d36 = Conv2d(400, 400, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=25, bias=False)

    def forward(self, x112):
        x113=self.batchnorm2d35(x112)
        x114=self.relu31(x113)
        x115=self.conv2d36(x114)
        return x115

m = M().eval()
x112 = torch.randn(torch.Size([1, 400, 14, 14]))
start = time.time()
output = m(x112)
end = time.time()
print(end-start)
