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
        self.conv2d16 = Conv2d(168, 168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=7, bias=False)
        self.batchnorm2d16 = BatchNorm2d(168, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x50):
        x51=self.conv2d16(x50)
        x52=self.batchnorm2d16(x51)
        return x52

m = M().eval()
x50 = torch.randn(torch.Size([1, 168, 28, 28]))
start = time.time()
output = m(x50)
end = time.time()
print(end-start)
