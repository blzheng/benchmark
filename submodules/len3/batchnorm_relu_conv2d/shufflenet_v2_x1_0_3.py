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
        self.batchnorm2d12 = BatchNorm2d(58, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu8 = ReLU(inplace=True)
        self.conv2d13 = Conv2d(58, 58, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=58, bias=False)

    def forward(self, x76):
        x77=self.batchnorm2d12(x76)
        x78=self.relu8(x77)
        x79=self.conv2d13(x78)
        return x79

m = M().eval()
x76 = torch.randn(torch.Size([1, 58, 28, 28]))
start = time.time()
output = m(x76)
end = time.time()
print(end-start)
