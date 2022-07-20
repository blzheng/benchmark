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
        self.relu49 = ReLU(inplace=True)
        self.conv2d66 = Conv2d(440, 440, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=55, bias=False)
        self.batchnorm2d42 = BatchNorm2d(440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x205):
        x206=self.relu49(x205)
        x207=self.conv2d66(x206)
        x208=self.batchnorm2d42(x207)
        return x208

m = M().eval()
x205 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x205)
end = time.time()
print(end-start)
