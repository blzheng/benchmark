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
        self.relu15 = ReLU(inplace=True)
        self.conv2d23 = Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
        self.batchnorm2d23 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x73):
        x74=self.relu15(x73)
        x75=self.conv2d23(x74)
        x76=self.batchnorm2d23(x75)
        return x76

m = M().eval()
x73 = torch.randn(torch.Size([1, 192, 28, 28]))
start = time.time()
output = m(x73)
end = time.time()
print(end-start)
