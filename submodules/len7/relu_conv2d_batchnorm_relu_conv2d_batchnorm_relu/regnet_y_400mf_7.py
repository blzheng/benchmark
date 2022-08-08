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
        self.relu44 = ReLU(inplace=True)
        self.conv2d60 = Conv2d(440, 440, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d38 = BatchNorm2d(440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu45 = ReLU(inplace=True)
        self.conv2d61 = Conv2d(440, 440, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=55, bias=False)
        self.batchnorm2d39 = BatchNorm2d(440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu46 = ReLU(inplace=True)

    def forward(self, x186):
        x187=self.relu44(x186)
        x188=self.conv2d60(x187)
        x189=self.batchnorm2d38(x188)
        x190=self.relu45(x189)
        x191=self.conv2d61(x190)
        x192=self.batchnorm2d39(x191)
        x193=self.relu46(x192)
        return x193

m = M().eval()
x186 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x186)
end = time.time()
print(end-start)