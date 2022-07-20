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
        self.conv2d99 = Conv2d(768, 768, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=768, bias=False)
        self.batchnorm2d59 = BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x296):
        x297=self.conv2d99(x296)
        x298=self.batchnorm2d59(x297)
        return x298

m = M().eval()
x296 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x296)
end = time.time()
print(end-start)
