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
        self.conv2d49 = Conv2d(400, 400, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d49 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu45 = ReLU(inplace=True)
        self.conv2d50 = Conv2d(400, 400, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x157, x151):
        x158=self.conv2d49(x157)
        x159=self.batchnorm2d49(x158)
        x160=operator.add(x151, x159)
        x161=self.relu45(x160)
        x162=self.conv2d50(x161)
        return x162

m = M().eval()
x157 = torch.randn(torch.Size([1, 400, 7, 7]))
x151 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x157, x151)
end = time.time()
print(end-start)
