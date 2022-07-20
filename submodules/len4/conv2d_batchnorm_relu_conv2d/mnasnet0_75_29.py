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
        self.conv2d43 = Conv2d(864, 864, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=864, bias=False)
        self.batchnorm2d43 = BatchNorm2d(864, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu29 = ReLU(inplace=True)
        self.conv2d44 = Conv2d(864, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x123):
        x124=self.conv2d43(x123)
        x125=self.batchnorm2d43(x124)
        x126=self.relu29(x125)
        x127=self.conv2d44(x126)
        return x127

m = M().eval()
x123 = torch.randn(torch.Size([1, 864, 7, 7]))
start = time.time()
output = m(x123)
end = time.time()
print(end-start)
