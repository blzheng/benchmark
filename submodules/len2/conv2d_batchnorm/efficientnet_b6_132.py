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
        self.conv2d222 = Conv2d(3456, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d132 = BatchNorm2d(576, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x696):
        x697=self.conv2d222(x696)
        x698=self.batchnorm2d132(x697)
        return x698

m = M().eval()
x696 = torch.randn(torch.Size([1, 3456, 7, 7]))
start = time.time()
output = m(x696)
end = time.time()
print(end-start)
