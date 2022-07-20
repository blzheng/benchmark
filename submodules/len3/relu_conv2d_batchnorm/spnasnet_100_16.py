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
        self.relu32 = ReLU(inplace=True)
        self.conv2d49 = Conv2d(576, 576, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=576, bias=False)
        self.batchnorm2d49 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x158):
        x159=self.relu32(x158)
        x160=self.conv2d49(x159)
        x161=self.batchnorm2d49(x160)
        return x161

m = M().eval()
x158 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x158)
end = time.time()
print(end-start)
