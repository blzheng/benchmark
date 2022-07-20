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
        self.batchnorm2d113 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu113 = ReLU(inplace=True)
        self.conv2d113 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x401):
        x402=self.batchnorm2d113(x401)
        x403=self.relu113(x402)
        x404=self.conv2d113(x403)
        return x404

m = M().eval()
x401 = torch.randn(torch.Size([1, 128, 7, 7]))
start = time.time()
output = m(x401)
end = time.time()
print(end-start)
