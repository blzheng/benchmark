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
        self.conv2d40 = Conv2d(240, 240, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=240, bias=False)
        self.batchnorm2d24 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x122):
        x123=self.conv2d40(x122)
        x124=self.batchnorm2d24(x123)
        return x124

m = M().eval()
x122 = torch.randn(torch.Size([1, 240, 28, 28]))
start = time.time()
output = m(x122)
end = time.time()
print(end-start)
