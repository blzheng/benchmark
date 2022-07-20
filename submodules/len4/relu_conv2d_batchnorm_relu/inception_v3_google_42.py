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
        self.conv2d78 = Conv2d(384, 384, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False)
        self.batchnorm2d78 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x266):
        x267=torch.nn.functional.relu(x266,inplace=True)
        x268=self.conv2d78(x267)
        x269=self.batchnorm2d78(x268)
        x270=torch.nn.functional.relu(x269,inplace=True)
        return x270

m = M().eval()
x266 = torch.randn(torch.Size([1, 384, 5, 5]))
start = time.time()
output = m(x266)
end = time.time()
print(end-start)