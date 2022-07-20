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
        self.conv2d20 = Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d20 = BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d21 = Conv2d(48, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)

    def forward(self, x76):
        x80=self.conv2d20(x76)
        x81=self.batchnorm2d20(x80)
        x82=torch.nn.functional.relu(x81,inplace=True)
        x83=self.conv2d21(x82)
        return x83

m = M().eval()
x76 = torch.randn(torch.Size([1, 288, 25, 25]))
start = time.time()
output = m(x76)
end = time.time()
print(end-start)
