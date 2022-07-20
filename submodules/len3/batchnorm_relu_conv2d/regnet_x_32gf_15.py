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
        self.batchnorm2d24 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu22 = ReLU(inplace=True)
        self.conv2d25 = Conv2d(672, 672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)

    def forward(self, x78):
        x79=self.batchnorm2d24(x78)
        x80=self.relu22(x79)
        x81=self.conv2d25(x80)
        return x81

m = M().eval()
x78 = torch.randn(torch.Size([1, 672, 28, 28]))
start = time.time()
output = m(x78)
end = time.time()
print(end-start)
