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
        self.batchnorm2d58 = BatchNorm2d(1104, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu39 = ReLU(inplace=True)
        self.conv2d59 = Conv2d(1104, 1104, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1104, bias=False)

    def forward(self, x189):
        x190=self.batchnorm2d58(x189)
        x191=self.relu39(x190)
        x192=self.conv2d59(x191)
        return x192

m = M().eval()
x189 = torch.randn(torch.Size([1, 1104, 7, 7]))
start = time.time()
output = m(x189)
end = time.time()
print(end-start)
