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
        self.conv2d20 = Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256, bias=False)
        self.batchnorm2d20 = BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x68):
        x69=self.conv2d20(x68)
        x70=self.batchnorm2d20(x69)
        return x70

m = M().eval()
x68 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x68)
end = time.time()
print(end-start)
