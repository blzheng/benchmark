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
        self.conv2d160 = Conv2d(1536, 1536, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1536, bias=False)
        self.batchnorm2d104 = BatchNorm2d(1536, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x510):
        x511=self.conv2d160(x510)
        x512=self.batchnorm2d104(x511)
        return x512

m = M().eval()
x510 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x510)
end = time.time()
print(end-start)
