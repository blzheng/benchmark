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
        self.conv2d188 = Conv2d(1344, 1344, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=1344, bias=False)
        self.batchnorm2d112 = BatchNorm2d(1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x563):
        x564=self.conv2d188(x563)
        x565=self.batchnorm2d112(x564)
        return x565

m = M().eval()
x563 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x563)
end = time.time()
print(end-start)
