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
        self.conv2d144 = Conv2d(1824, 1824, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1824, bias=False)
        self.batchnorm2d86 = BatchNorm2d(1824, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x449):
        x450=self.conv2d144(x449)
        x451=self.batchnorm2d86(x450)
        return x451

m = M().eval()
x449 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x449)
end = time.time()
print(end-start)