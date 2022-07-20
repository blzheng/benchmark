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
        self.conv2d149 = Conv2d(1200, 1200, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1200, bias=False)
        self.batchnorm2d89 = BatchNorm2d(1200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x445):
        x446=self.conv2d149(x445)
        x447=self.batchnorm2d89(x446)
        return x447

m = M().eval()
x445 = torch.randn(torch.Size([1, 1200, 14, 14]))
start = time.time()
output = m(x445)
end = time.time()
print(end-start)
