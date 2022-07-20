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
        self.batchnorm2d46 = BatchNorm2d(352, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu30 = ReLU(inplace=True)
        self.conv2d47 = Conv2d(352, 352, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=352, bias=False)

    def forward(self, x300):
        x301=self.batchnorm2d46(x300)
        x302=self.relu30(x301)
        x303=self.conv2d47(x302)
        return x303

m = M().eval()
x300 = torch.randn(torch.Size([1, 352, 7, 7]))
start = time.time()
output = m(x300)
end = time.time()
print(end-start)
