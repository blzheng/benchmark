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
        self.conv2d30 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False)
        self.batchnorm2d26 = BatchNorm2d(512, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x98):
        x99=self.conv2d30(x98)
        x100=self.batchnorm2d26(x99)
        return x100

m = M().eval()
x98 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x98)
end = time.time()
print(end-start)
