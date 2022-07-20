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
        self.conv2d17 = Conv2d(288, 288, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=18, bias=False)
        self.batchnorm2d17 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x52):
        x53=self.conv2d17(x52)
        x54=self.batchnorm2d17(x53)
        return x54

m = M().eval()
x52 = torch.randn(torch.Size([1, 288, 28, 28]))
start = time.time()
output = m(x52)
end = time.time()
print(end-start)