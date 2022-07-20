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
        self.conv2d60 = Conv2d(528, 528, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=528, bias=False)
        self.batchnorm2d36 = BatchNorm2d(528, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x178):
        x179=self.conv2d60(x178)
        x180=self.batchnorm2d36(x179)
        return x180

m = M().eval()
x178 = torch.randn(torch.Size([1, 528, 14, 14]))
start = time.time()
output = m(x178)
end = time.time()
print(end-start)
