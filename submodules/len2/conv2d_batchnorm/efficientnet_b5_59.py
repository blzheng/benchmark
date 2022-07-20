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
        self.batchnorm2d59 = BatchNorm2d(768, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x309):
        x310=self.conv2d99(x309)
        x311=self.batchnorm2d59(x310)
        return x311

m = M().eval()
x309 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x309)
end = time.time()
print(end-start)