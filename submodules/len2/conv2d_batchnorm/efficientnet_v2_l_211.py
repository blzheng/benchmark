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
        self.conv2d329 = Conv2d(3840, 3840, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=3840, bias=False)
        self.batchnorm2d211 = BatchNorm2d(3840, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x1056):
        x1057=self.conv2d329(x1056)
        x1058=self.batchnorm2d211(x1057)
        return x1058

m = M().eval()
x1056 = torch.randn(torch.Size([1, 3840, 7, 7]))
start = time.time()
output = m(x1056)
end = time.time()
print(end-start)
