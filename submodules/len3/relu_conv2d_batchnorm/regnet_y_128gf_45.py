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
        self.relu89 = ReLU(inplace=True)
        self.conv2d115 = Conv2d(2904, 2904, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=11, bias=False)
        self.batchnorm2d71 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x363):
        x364=self.relu89(x363)
        x365=self.conv2d115(x364)
        x366=self.batchnorm2d71(x365)
        return x366

m = M().eval()
x363 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x363)
end = time.time()
print(end-start)
