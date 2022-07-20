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
        self.conv2d99 = Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
        self.batchnorm2d99 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu94 = ReLU(inplace=True)

    def forward(self, x325):
        x326=self.conv2d99(x325)
        x327=self.batchnorm2d99(x326)
        x328=self.relu94(x327)
        return x328

m = M().eval()
x325 = torch.randn(torch.Size([1, 2048, 7, 7]))
start = time.time()
output = m(x325)
end = time.time()
print(end-start)
