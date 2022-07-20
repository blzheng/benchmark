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
        self.conv2d153 = Conv2d(1344, 1344, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1344, bias=False)
        self.batchnorm2d91 = BatchNorm2d(1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x458):
        x459=self.conv2d153(x458)
        x460=self.batchnorm2d91(x459)
        return x460

m = M().eval()
x458 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x458)
end = time.time()
print(end-start)
