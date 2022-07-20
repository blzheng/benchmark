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
        self.conv2d35 = Conv2d(384, 384, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=384, bias=False)
        self.batchnorm2d35 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu24 = ReLU(inplace=True)

    def forward(self, x113):
        x114=self.conv2d35(x113)
        x115=self.batchnorm2d35(x114)
        x116=self.relu24(x115)
        return x116

m = M().eval()
x113 = torch.randn(torch.Size([1, 384, 14, 14]))
start = time.time()
output = m(x113)
end = time.time()
print(end-start)
