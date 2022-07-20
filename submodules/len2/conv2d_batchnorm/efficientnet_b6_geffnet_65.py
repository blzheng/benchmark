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
        self.conv2d109 = Conv2d(864, 864, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=864, bias=False)
        self.batchnorm2d65 = BatchNorm2d(864, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x326):
        x327=self.conv2d109(x326)
        x328=self.batchnorm2d65(x327)
        return x328

m = M().eval()
x326 = torch.randn(torch.Size([1, 864, 14, 14]))
start = time.time()
output = m(x326)
end = time.time()
print(end-start)
