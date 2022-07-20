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
        self.conv2d105 = Conv2d(960, 960, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=960, bias=False)
        self.batchnorm2d63 = BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x326):
        x327=self.conv2d105(x326)
        x328=self.batchnorm2d63(x327)
        return x328

m = M().eval()
x326 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x326)
end = time.time()
print(end-start)
