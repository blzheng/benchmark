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
        self.conv2d159 = Conv2d(2064, 2064, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2064, bias=False)
        self.batchnorm2d95 = BatchNorm2d(2064, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x497):
        x498=self.conv2d159(x497)
        x499=self.batchnorm2d95(x498)
        return x499

m = M().eval()
x497 = torch.randn(torch.Size([1, 2064, 7, 7]))
start = time.time()
output = m(x497)
end = time.time()
print(end-start)
