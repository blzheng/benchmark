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
        self.conv2d90 = Conv2d(816, 816, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=816, bias=False)
        self.batchnorm2d54 = BatchNorm2d(816, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x267):
        x268=self.conv2d90(x267)
        x269=self.batchnorm2d54(x268)
        return x269

m = M().eval()
x267 = torch.randn(torch.Size([1, 816, 14, 14]))
start = time.time()
output = m(x267)
end = time.time()
print(end-start)
