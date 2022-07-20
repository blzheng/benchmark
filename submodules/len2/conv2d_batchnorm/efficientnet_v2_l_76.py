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
        self.conv2d104 = Conv2d(1344, 1344, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1344, bias=False)
        self.batchnorm2d76 = BatchNorm2d(1344, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x340):
        x341=self.conv2d104(x340)
        x342=self.batchnorm2d76(x341)
        return x342

m = M().eval()
x340 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x340)
end = time.time()
print(end-start)
