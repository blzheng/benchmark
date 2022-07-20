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
        self.conv2d53 = Conv2d(288, 288, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=288, bias=False)
        self.batchnorm2d31 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x161):
        x162=self.conv2d53(x161)
        x163=self.batchnorm2d31(x162)
        return x163

m = M().eval()
x161 = torch.randn(torch.Size([1, 288, 56, 56]))
start = time.time()
output = m(x161)
end = time.time()
print(end-start)
