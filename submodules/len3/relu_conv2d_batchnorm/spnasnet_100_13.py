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
        self.relu26 = ReLU(inplace=True)
        self.conv2d40 = Conv2d(288, 288, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=288, bias=False)
        self.batchnorm2d40 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x128):
        x129=self.relu26(x128)
        x130=self.conv2d40(x129)
        x131=self.batchnorm2d40(x130)
        return x131

m = M().eval()
x128 = torch.randn(torch.Size([1, 288, 14, 14]))
start = time.time()
output = m(x128)
end = time.time()
print(end-start)
