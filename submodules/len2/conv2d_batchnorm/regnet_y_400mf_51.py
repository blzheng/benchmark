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
        self.conv2d81 = Conv2d(440, 440, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=55, bias=False)
        self.batchnorm2d51 = BatchNorm2d(440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x254):
        x255=self.conv2d81(x254)
        x256=self.batchnorm2d51(x255)
        return x256

m = M().eval()
x254 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x254)
end = time.time()
print(end-start)
