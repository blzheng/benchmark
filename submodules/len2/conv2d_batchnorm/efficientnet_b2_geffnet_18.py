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
        self.conv2d30 = Conv2d(288, 288, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=288, bias=False)
        self.batchnorm2d18 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x89):
        x90=self.conv2d30(x89)
        x91=self.batchnorm2d18(x90)
        return x91

m = M().eval()
x89 = torch.randn(torch.Size([1, 288, 28, 28]))
start = time.time()
output = m(x89)
end = time.time()
print(end-start)
