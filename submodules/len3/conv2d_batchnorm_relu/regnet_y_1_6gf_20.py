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
        self.conv2d50 = Conv2d(336, 336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=14, bias=False)
        self.batchnorm2d32 = BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu38 = ReLU(inplace=True)

    def forward(self, x156):
        x157=self.conv2d50(x156)
        x158=self.batchnorm2d32(x157)
        x159=self.relu38(x158)
        return x159

m = M().eval()
x156 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x156)
end = time.time()
print(end-start)
