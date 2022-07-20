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
        self.conv2d59 = Conv2d(432, 432, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=432, bias=False)
        self.batchnorm2d35 = BatchNorm2d(432, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x183):
        x184=self.conv2d59(x183)
        x185=self.batchnorm2d35(x184)
        return x185

m = M().eval()
x183 = torch.randn(torch.Size([1, 432, 28, 28]))
start = time.time()
output = m(x183)
end = time.time()
print(end-start)
