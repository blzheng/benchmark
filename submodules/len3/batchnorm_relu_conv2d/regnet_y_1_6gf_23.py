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
        self.batchnorm2d70 = BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu89 = ReLU(inplace=True)
        self.conv2d115 = Conv2d(336, 336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=14, bias=False)

    def forward(self, x362):
        x363=self.batchnorm2d70(x362)
        x364=self.relu89(x363)
        x365=self.conv2d115(x364)
        return x365

m = M().eval()
x362 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x362)
end = time.time()
print(end-start)