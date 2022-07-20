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
        self.relu68 = ReLU6(inplace=True)
        self.conv2d13 = Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
        self.batchnorm2d13 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x35):
        x36=self.relu68(x35)
        x37=self.conv2d13(x36)
        x38=self.batchnorm2d13(x37)
        return x38

m = M().eval()
x35 = torch.randn(torch.Size([1, 192, 28, 28]))
start = time.time()
output = m(x35)
end = time.time()
print(end-start)
