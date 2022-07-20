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
        self.relu21 = ReLU(inplace=True)
        self.conv2d30 = Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=20, bias=False)
        self.batchnorm2d20 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x91):
        x92=self.relu21(x91)
        x93=self.conv2d30(x92)
        x94=self.batchnorm2d20(x93)
        return x94

m = M().eval()
x91 = torch.randn(torch.Size([1, 320, 14, 14]))
start = time.time()
output = m(x91)
end = time.time()
print(end-start)
