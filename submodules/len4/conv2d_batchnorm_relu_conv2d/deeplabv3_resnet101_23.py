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
        self.conv2d38 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d38 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu34 = ReLU(inplace=True)
        self.conv2d39 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x125):
        x126=self.conv2d38(x125)
        x127=self.batchnorm2d38(x126)
        x128=self.relu34(x127)
        x129=self.conv2d39(x128)
        return x129

m = M().eval()
x125 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x125)
end = time.time()
print(end-start)
