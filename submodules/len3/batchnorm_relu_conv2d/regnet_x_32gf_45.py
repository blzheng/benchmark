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
        self.batchnorm2d71 = BatchNorm2d(2520, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu67 = ReLU(inplace=True)
        self.conv2d72 = Conv2d(2520, 2520, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=15, bias=False)

    def forward(self, x232):
        x233=self.batchnorm2d71(x232)
        x234=self.relu67(x233)
        x235=self.conv2d72(x234)
        return x235

m = M().eval()
x232 = torch.randn(torch.Size([1, 2520, 14, 14]))
start = time.time()
output = m(x232)
end = time.time()
print(end-start)
