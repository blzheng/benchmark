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
        self.batchnorm2d172 = BatchNorm2d(1472, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu172 = ReLU(inplace=True)
        self.conv2d172 = Conv2d(1472, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x608):
        x609=self.batchnorm2d172(x608)
        x610=self.relu172(x609)
        x611=self.conv2d172(x610)
        return x611

m = M().eval()
x608 = torch.randn(torch.Size([1, 1472, 7, 7]))
start = time.time()
output = m(x608)
end = time.time()
print(end-start)
