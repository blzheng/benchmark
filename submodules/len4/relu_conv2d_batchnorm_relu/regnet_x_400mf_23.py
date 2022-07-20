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
        self.relu40 = ReLU(inplace=True)
        self.conv2d45 = Conv2d(400, 400, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=25, bias=False)
        self.batchnorm2d45 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu41 = ReLU(inplace=True)

    def forward(self, x143):
        x144=self.relu40(x143)
        x145=self.conv2d45(x144)
        x146=self.batchnorm2d45(x145)
        x147=self.relu41(x146)
        return x147

m = M().eval()
x143 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x143)
end = time.time()
print(end-start)
