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
        self.conv2d40 = Conv2d(576, 576, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=24, bias=False)
        self.batchnorm2d26 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x124):
        x125=self.conv2d40(x124)
        x126=self.batchnorm2d26(x125)
        return x126

m = M().eval()
x124 = torch.randn(torch.Size([1, 576, 28, 28]))
start = time.time()
output = m(x124)
end = time.time()
print(end-start)
