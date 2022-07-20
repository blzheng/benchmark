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
        self.relu30 = ReLU(inplace=True)
        self.conv2d47 = Conv2d(488, 488, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=488, bias=False)
        self.batchnorm2d47 = BatchNorm2d(488, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d48 = Conv2d(488, 488, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x301):
        x302=self.relu30(x301)
        x303=self.conv2d47(x302)
        x304=self.batchnorm2d47(x303)
        x305=self.conv2d48(x304)
        return x305

m = M().eval()
x301 = torch.randn(torch.Size([1, 488, 7, 7]))
start = time.time()
output = m(x301)
end = time.time()
print(end-start)
