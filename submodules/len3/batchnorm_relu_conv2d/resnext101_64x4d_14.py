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
        self.batchnorm2d24 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu22 = ReLU(inplace=True)
        self.conv2d25 = Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=64, bias=False)

    def forward(self, x79):
        x80=self.batchnorm2d24(x79)
        x81=self.relu22(x80)
        x82=self.conv2d25(x81)
        return x82

m = M().eval()
x79 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x79)
end = time.time()
print(end-start)
