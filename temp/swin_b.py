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
import sys
import os

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.conv2d0 = Conv2d(3, 128, kernel_size=(4, 4), stride=(4, 4))
        self.layernorm0 = LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        self.layernorm1 = LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        self.layernorm2 = LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        self.linear0 = Linear(in_features=128, out_features=512, bias=True)
        self.gelu0 = GELU(approximate='none')
        self.dropout0 = Dropout(p=0.0, inplace=False)
        self.linear1 = Linear(in_features=512, out_features=128, bias=True)
        self.dropout1 = Dropout(p=0.0, inplace=False)
        self.layernorm3 = LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        self.layernorm4 = LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        self.linear2 = Linear(in_features=128, out_features=512, bias=True)
        self.gelu1 = GELU(approximate='none')
        self.dropout2 = Dropout(p=0.0, inplace=False)
        self.linear3 = Linear(in_features=512, out_features=128, bias=True)
        self.dropout3 = Dropout(p=0.0, inplace=False)
        self.layernorm5 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear4 = Linear(in_features=512, out_features=256, bias=False)
        self.layernorm6 = LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        self.layernorm7 = LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        self.linear5 = Linear(in_features=256, out_features=1024, bias=True)
        self.gelu2 = GELU(approximate='none')
        self.dropout4 = Dropout(p=0.0, inplace=False)
        self.linear6 = Linear(in_features=1024, out_features=256, bias=True)
        self.dropout5 = Dropout(p=0.0, inplace=False)
        self.layernorm8 = LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        self.layernorm9 = LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        self.linear7 = Linear(in_features=256, out_features=1024, bias=True)
        self.gelu3 = GELU(approximate='none')
        self.dropout6 = Dropout(p=0.0, inplace=False)
        self.linear8 = Linear(in_features=1024, out_features=256, bias=True)
        self.dropout7 = Dropout(p=0.0, inplace=False)
        self.layernorm10 = LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        self.linear9 = Linear(in_features=1024, out_features=512, bias=False)
        self.layernorm11 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm12 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear10 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu4 = GELU(approximate='none')
        self.dropout8 = Dropout(p=0.0, inplace=False)
        self.linear11 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout9 = Dropout(p=0.0, inplace=False)
        self.layernorm13 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm14 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear12 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu5 = GELU(approximate='none')
        self.dropout10 = Dropout(p=0.0, inplace=False)
        self.linear13 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout11 = Dropout(p=0.0, inplace=False)
        self.layernorm15 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm16 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear14 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu6 = GELU(approximate='none')
        self.dropout12 = Dropout(p=0.0, inplace=False)
        self.linear15 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout13 = Dropout(p=0.0, inplace=False)
        self.layernorm17 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm18 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear16 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu7 = GELU(approximate='none')
        self.dropout14 = Dropout(p=0.0, inplace=False)
        self.linear17 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout15 = Dropout(p=0.0, inplace=False)
        self.layernorm19 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm20 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear18 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu8 = GELU(approximate='none')
        self.dropout16 = Dropout(p=0.0, inplace=False)
        self.linear19 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout17 = Dropout(p=0.0, inplace=False)
        self.layernorm21 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm22 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear20 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu9 = GELU(approximate='none')
        self.dropout18 = Dropout(p=0.0, inplace=False)
        self.linear21 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout19 = Dropout(p=0.0, inplace=False)
        self.layernorm23 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm24 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear22 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu10 = GELU(approximate='none')
        self.dropout20 = Dropout(p=0.0, inplace=False)
        self.linear23 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout21 = Dropout(p=0.0, inplace=False)
        self.layernorm25 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm26 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear24 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu11 = GELU(approximate='none')
        self.dropout22 = Dropout(p=0.0, inplace=False)
        self.linear25 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout23 = Dropout(p=0.0, inplace=False)
        self.layernorm27 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm28 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear26 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu12 = GELU(approximate='none')
        self.dropout24 = Dropout(p=0.0, inplace=False)
        self.linear27 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout25 = Dropout(p=0.0, inplace=False)
        self.layernorm29 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm30 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear28 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu13 = GELU(approximate='none')
        self.dropout26 = Dropout(p=0.0, inplace=False)
        self.linear29 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout27 = Dropout(p=0.0, inplace=False)
        self.layernorm31 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm32 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear30 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu14 = GELU(approximate='none')
        self.dropout28 = Dropout(p=0.0, inplace=False)
        self.linear31 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout29 = Dropout(p=0.0, inplace=False)
        self.layernorm33 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm34 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear32 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu15 = GELU(approximate='none')
        self.dropout30 = Dropout(p=0.0, inplace=False)
        self.linear33 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout31 = Dropout(p=0.0, inplace=False)
        self.layernorm35 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm36 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear34 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu16 = GELU(approximate='none')
        self.dropout32 = Dropout(p=0.0, inplace=False)
        self.linear35 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout33 = Dropout(p=0.0, inplace=False)
        self.layernorm37 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm38 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear36 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu17 = GELU(approximate='none')
        self.dropout34 = Dropout(p=0.0, inplace=False)
        self.linear37 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout35 = Dropout(p=0.0, inplace=False)
        self.layernorm39 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm40 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear38 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu18 = GELU(approximate='none')
        self.dropout36 = Dropout(p=0.0, inplace=False)
        self.linear39 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout37 = Dropout(p=0.0, inplace=False)
        self.layernorm41 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm42 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear40 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu19 = GELU(approximate='none')
        self.dropout38 = Dropout(p=0.0, inplace=False)
        self.linear41 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout39 = Dropout(p=0.0, inplace=False)
        self.layernorm43 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm44 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear42 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu20 = GELU(approximate='none')
        self.dropout40 = Dropout(p=0.0, inplace=False)
        self.linear43 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout41 = Dropout(p=0.0, inplace=False)
        self.layernorm45 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.layernorm46 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear44 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu21 = GELU(approximate='none')
        self.dropout42 = Dropout(p=0.0, inplace=False)
        self.linear45 = Linear(in_features=2048, out_features=512, bias=True)
        self.dropout43 = Dropout(p=0.0, inplace=False)
        self.layernorm47 = LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
        self.linear46 = Linear(in_features=2048, out_features=1024, bias=False)
        self.layernorm48 = LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        self.layernorm49 = LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        self.linear47 = Linear(in_features=1024, out_features=4096, bias=True)
        self.gelu22 = GELU(approximate='none')
        self.dropout44 = Dropout(p=0.0, inplace=False)
        self.linear48 = Linear(in_features=4096, out_features=1024, bias=True)
        self.dropout45 = Dropout(p=0.0, inplace=False)
        self.layernorm50 = LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        self.layernorm51 = LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        self.linear49 = Linear(in_features=1024, out_features=4096, bias=True)
        self.gelu23 = GELU(approximate='none')
        self.dropout46 = Dropout(p=0.0, inplace=False)
        self.linear50 = Linear(in_features=4096, out_features=1024, bias=True)
        self.dropout47 = Dropout(p=0.0, inplace=False)
        self.layernorm52 = LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        self.adaptiveavgpool2d0 = AdaptiveAvgPool2d(output_size=1)
        self.linear51 = Linear(in_features=1024, out_features=1000, bias=True)
        self.relative_position_bias_table0 = torch.rand(torch.Size([169, 4])).to(torch.float32)
        self.relative_position_index0 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight0 = torch.rand(torch.Size([384, 128])).to(torch.float32)
        self.weight1 = torch.rand(torch.Size([128, 128])).to(torch.float32)
        self.bias0 = torch.rand(torch.Size([384])).to(torch.float32)
        self.bias1 = torch.rand(torch.Size([128])).to(torch.float32)
        self.relative_position_bias_table1 = torch.rand(torch.Size([169, 4])).to(torch.float32)
        self.relative_position_index1 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight2 = torch.rand(torch.Size([384, 128])).to(torch.float32)
        self.weight3 = torch.rand(torch.Size([128, 128])).to(torch.float32)
        self.bias2 = torch.rand(torch.Size([384])).to(torch.float32)
        self.bias3 = torch.rand(torch.Size([128])).to(torch.float32)
        self.relative_position_bias_table2 = torch.rand(torch.Size([169, 8])).to(torch.float32)
        self.relative_position_index2 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight4 = torch.rand(torch.Size([768, 256])).to(torch.float32)
        self.weight5 = torch.rand(torch.Size([256, 256])).to(torch.float32)
        self.bias4 = torch.rand(torch.Size([768])).to(torch.float32)
        self.bias5 = torch.rand(torch.Size([256])).to(torch.float32)
        self.relative_position_bias_table3 = torch.rand(torch.Size([169, 8])).to(torch.float32)
        self.relative_position_index3 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight6 = torch.rand(torch.Size([768, 256])).to(torch.float32)
        self.weight7 = torch.rand(torch.Size([256, 256])).to(torch.float32)
        self.bias6 = torch.rand(torch.Size([768])).to(torch.float32)
        self.bias7 = torch.rand(torch.Size([256])).to(torch.float32)
        self.relative_position_bias_table4 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index4 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight8 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight9 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias8 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias9 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table5 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index5 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight10 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight11 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias10 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias11 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table6 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index6 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight12 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight13 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias12 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias13 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table7 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index7 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight14 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight15 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias14 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias15 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table8 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index8 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight16 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight17 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias16 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias17 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table9 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index9 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight18 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight19 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias18 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias19 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table10 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index10 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight20 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight21 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias20 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias21 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table11 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index11 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight22 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight23 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias22 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias23 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table12 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index12 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight24 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight25 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias24 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias25 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table13 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index13 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight26 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight27 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias26 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias27 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table14 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index14 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight28 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight29 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias28 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias29 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table15 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index15 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight30 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight31 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias30 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias31 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table16 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index16 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight32 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight33 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias32 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias33 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table17 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index17 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight34 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight35 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias34 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias35 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table18 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index18 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight36 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight37 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias36 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias37 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table19 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index19 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight38 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight39 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias38 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias39 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table20 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index20 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight40 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight41 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias40 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias41 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table21 = torch.rand(torch.Size([169, 16])).to(torch.float32)
        self.relative_position_index21 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight42 = torch.rand(torch.Size([1536, 512])).to(torch.float32)
        self.weight43 = torch.rand(torch.Size([512, 512])).to(torch.float32)
        self.bias42 = torch.rand(torch.Size([1536])).to(torch.float32)
        self.bias43 = torch.rand(torch.Size([512])).to(torch.float32)
        self.relative_position_bias_table22 = torch.rand(torch.Size([169, 32])).to(torch.float32)
        self.relative_position_index22 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight44 = torch.rand(torch.Size([3072, 1024])).to(torch.float32)
        self.weight45 = torch.rand(torch.Size([1024, 1024])).to(torch.float32)
        self.bias44 = torch.rand(torch.Size([3072])).to(torch.float32)
        self.bias45 = torch.rand(torch.Size([1024])).to(torch.float32)
        self.relative_position_bias_table23 = torch.rand(torch.Size([169, 32])).to(torch.float32)
        self.relative_position_index23 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight46 = torch.rand(torch.Size([3072, 1024])).to(torch.float32)
        self.weight47 = torch.rand(torch.Size([1024, 1024])).to(torch.float32)
        self.bias46 = torch.rand(torch.Size([3072])).to(torch.float32)
        self.bias47 = torch.rand(torch.Size([1024])).to(torch.float32)

    def forward(self, x):
        x0=x
        if x0 is None:
            print('x0: {}'.format(x0))
        elif isinstance(x0, torch.Tensor):
            print('x0: {}'.format(x0.shape))
        elif isinstance(x0, tuple):
            tuple_shapes = '('
            for item in x0:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x0: {}'.format(tuple_shapes))
        else:
            print('x0: {}'.format(x0))
        x1=self.conv2d0(x0)
        if x1 is None:
            print('x1: {}'.format(x1))
        elif isinstance(x1, torch.Tensor):
            print('x1: {}'.format(x1.shape))
        elif isinstance(x1, tuple):
            tuple_shapes = '('
            for item in x1:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x1: {}'.format(tuple_shapes))
        else:
            print('x1: {}'.format(x1))
        x2=torch.permute(x1, [0, 2, 3, 1])
        if x2 is None:
            print('x2: {}'.format(x2))
        elif isinstance(x2, torch.Tensor):
            print('x2: {}'.format(x2.shape))
        elif isinstance(x2, tuple):
            tuple_shapes = '('
            for item in x2:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x2: {}'.format(tuple_shapes))
        else:
            print('x2: {}'.format(x2))
        x3=self.layernorm0(x2)
        if x3 is None:
            print('x3: {}'.format(x3))
        elif isinstance(x3, torch.Tensor):
            print('x3: {}'.format(x3.shape))
        elif isinstance(x3, tuple):
            tuple_shapes = '('
            for item in x3:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x3: {}'.format(tuple_shapes))
        else:
            print('x3: {}'.format(x3))
        x4=self.layernorm1(x3)
        if x4 is None:
            print('x4: {}'.format(x4))
        elif isinstance(x4, torch.Tensor):
            print('x4: {}'.format(x4.shape))
        elif isinstance(x4, tuple):
            tuple_shapes = '('
            for item in x4:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x4: {}'.format(tuple_shapes))
        else:
            print('x4: {}'.format(x4))
        x7=operator.getitem(self.relative_position_bias_table0, self.relative_position_index0)
        if x7 is None:
            print('x7: {}'.format(x7))
        elif isinstance(x7, torch.Tensor):
            print('x7: {}'.format(x7.shape))
        elif isinstance(x7, tuple):
            tuple_shapes = '('
            for item in x7:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x7: {}'.format(tuple_shapes))
        else:
            print('x7: {}'.format(x7))
        x8=x7.view(49, 49, -1)
        if x8 is None:
            print('x8: {}'.format(x8))
        elif isinstance(x8, torch.Tensor):
            print('x8: {}'.format(x8.shape))
        elif isinstance(x8, tuple):
            tuple_shapes = '('
            for item in x8:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x8: {}'.format(tuple_shapes))
        else:
            print('x8: {}'.format(x8))
        x9=x8.permute(2, 0, 1)
        if x9 is None:
            print('x9: {}'.format(x9))
        elif isinstance(x9, torch.Tensor):
            print('x9: {}'.format(x9.shape))
        elif isinstance(x9, tuple):
            tuple_shapes = '('
            for item in x9:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x9: {}'.format(tuple_shapes))
        else:
            print('x9: {}'.format(x9))
        x10=x9.contiguous()
        if x10 is None:
            print('x10: {}'.format(x10))
        elif isinstance(x10, torch.Tensor):
            print('x10: {}'.format(x10.shape))
        elif isinstance(x10, tuple):
            tuple_shapes = '('
            for item in x10:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x10: {}'.format(tuple_shapes))
        else:
            print('x10: {}'.format(x10))
        x11=x10.unsqueeze(0)
        if x11 is None:
            print('x11: {}'.format(x11))
        elif isinstance(x11, torch.Tensor):
            print('x11: {}'.format(x11.shape))
        elif isinstance(x11, tuple):
            tuple_shapes = '('
            for item in x11:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x11: {}'.format(tuple_shapes))
        else:
            print('x11: {}'.format(x11))
        x16=torchvision.models.swin_transformer.shifted_window_attention(x4, self.weight0, self.weight1, x11, [7, 7], 4,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias0, proj_bias=self.bias1)
        if x16 is None:
            print('x16: {}'.format(x16))
        elif isinstance(x16, torch.Tensor):
            print('x16: {}'.format(x16.shape))
        elif isinstance(x16, tuple):
            tuple_shapes = '('
            for item in x16:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x16: {}'.format(tuple_shapes))
        else:
            print('x16: {}'.format(x16))
        x17=stochastic_depth(x16, 0.0, 'row', False)
        if x17 is None:
            print('x17: {}'.format(x17))
        elif isinstance(x17, torch.Tensor):
            print('x17: {}'.format(x17.shape))
        elif isinstance(x17, tuple):
            tuple_shapes = '('
            for item in x17:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x17: {}'.format(tuple_shapes))
        else:
            print('x17: {}'.format(x17))
        x18=operator.add(x3, x17)
        if x18 is None:
            print('x18: {}'.format(x18))
        elif isinstance(x18, torch.Tensor):
            print('x18: {}'.format(x18.shape))
        elif isinstance(x18, tuple):
            tuple_shapes = '('
            for item in x18:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x18: {}'.format(tuple_shapes))
        else:
            print('x18: {}'.format(x18))
        x19=self.layernorm2(x18)
        if x19 is None:
            print('x19: {}'.format(x19))
        elif isinstance(x19, torch.Tensor):
            print('x19: {}'.format(x19.shape))
        elif isinstance(x19, tuple):
            tuple_shapes = '('
            for item in x19:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x19: {}'.format(tuple_shapes))
        else:
            print('x19: {}'.format(x19))
        x20=self.linear0(x19)
        if x20 is None:
            print('x20: {}'.format(x20))
        elif isinstance(x20, torch.Tensor):
            print('x20: {}'.format(x20.shape))
        elif isinstance(x20, tuple):
            tuple_shapes = '('
            for item in x20:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x20: {}'.format(tuple_shapes))
        else:
            print('x20: {}'.format(x20))
        x21=self.gelu0(x20)
        if x21 is None:
            print('x21: {}'.format(x21))
        elif isinstance(x21, torch.Tensor):
            print('x21: {}'.format(x21.shape))
        elif isinstance(x21, tuple):
            tuple_shapes = '('
            for item in x21:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x21: {}'.format(tuple_shapes))
        else:
            print('x21: {}'.format(x21))
        x22=self.dropout0(x21)
        if x22 is None:
            print('x22: {}'.format(x22))
        elif isinstance(x22, torch.Tensor):
            print('x22: {}'.format(x22.shape))
        elif isinstance(x22, tuple):
            tuple_shapes = '('
            for item in x22:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x22: {}'.format(tuple_shapes))
        else:
            print('x22: {}'.format(x22))
        x23=self.linear1(x22)
        if x23 is None:
            print('x23: {}'.format(x23))
        elif isinstance(x23, torch.Tensor):
            print('x23: {}'.format(x23.shape))
        elif isinstance(x23, tuple):
            tuple_shapes = '('
            for item in x23:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x23: {}'.format(tuple_shapes))
        else:
            print('x23: {}'.format(x23))
        x24=self.dropout1(x23)
        if x24 is None:
            print('x24: {}'.format(x24))
        elif isinstance(x24, torch.Tensor):
            print('x24: {}'.format(x24.shape))
        elif isinstance(x24, tuple):
            tuple_shapes = '('
            for item in x24:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x24: {}'.format(tuple_shapes))
        else:
            print('x24: {}'.format(x24))
        x25=stochastic_depth(x24, 0.0, 'row', False)
        if x25 is None:
            print('x25: {}'.format(x25))
        elif isinstance(x25, torch.Tensor):
            print('x25: {}'.format(x25.shape))
        elif isinstance(x25, tuple):
            tuple_shapes = '('
            for item in x25:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x25: {}'.format(tuple_shapes))
        else:
            print('x25: {}'.format(x25))
        x26=operator.add(x18, x25)
        if x26 is None:
            print('x26: {}'.format(x26))
        elif isinstance(x26, torch.Tensor):
            print('x26: {}'.format(x26.shape))
        elif isinstance(x26, tuple):
            tuple_shapes = '('
            for item in x26:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x26: {}'.format(tuple_shapes))
        else:
            print('x26: {}'.format(x26))
        x27=self.layernorm3(x26)
        if x27 is None:
            print('x27: {}'.format(x27))
        elif isinstance(x27, torch.Tensor):
            print('x27: {}'.format(x27.shape))
        elif isinstance(x27, tuple):
            tuple_shapes = '('
            for item in x27:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x27: {}'.format(tuple_shapes))
        else:
            print('x27: {}'.format(x27))
        x30=operator.getitem(self.relative_position_bias_table1, self.relative_position_index1)
        if x30 is None:
            print('x30: {}'.format(x30))
        elif isinstance(x30, torch.Tensor):
            print('x30: {}'.format(x30.shape))
        elif isinstance(x30, tuple):
            tuple_shapes = '('
            for item in x30:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x30: {}'.format(tuple_shapes))
        else:
            print('x30: {}'.format(x30))
        x31=x30.view(49, 49, -1)
        if x31 is None:
            print('x31: {}'.format(x31))
        elif isinstance(x31, torch.Tensor):
            print('x31: {}'.format(x31.shape))
        elif isinstance(x31, tuple):
            tuple_shapes = '('
            for item in x31:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x31: {}'.format(tuple_shapes))
        else:
            print('x31: {}'.format(x31))
        x32=x31.permute(2, 0, 1)
        if x32 is None:
            print('x32: {}'.format(x32))
        elif isinstance(x32, torch.Tensor):
            print('x32: {}'.format(x32.shape))
        elif isinstance(x32, tuple):
            tuple_shapes = '('
            for item in x32:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x32: {}'.format(tuple_shapes))
        else:
            print('x32: {}'.format(x32))
        x33=x32.contiguous()
        if x33 is None:
            print('x33: {}'.format(x33))
        elif isinstance(x33, torch.Tensor):
            print('x33: {}'.format(x33.shape))
        elif isinstance(x33, tuple):
            tuple_shapes = '('
            for item in x33:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x33: {}'.format(tuple_shapes))
        else:
            print('x33: {}'.format(x33))
        x34=x33.unsqueeze(0)
        if x34 is None:
            print('x34: {}'.format(x34))
        elif isinstance(x34, torch.Tensor):
            print('x34: {}'.format(x34.shape))
        elif isinstance(x34, tuple):
            tuple_shapes = '('
            for item in x34:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x34: {}'.format(tuple_shapes))
        else:
            print('x34: {}'.format(x34))
        x39=torchvision.models.swin_transformer.shifted_window_attention(x27, self.weight2, self.weight3, x34, [7, 7], 4,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias2, proj_bias=self.bias3)
        if x39 is None:
            print('x39: {}'.format(x39))
        elif isinstance(x39, torch.Tensor):
            print('x39: {}'.format(x39.shape))
        elif isinstance(x39, tuple):
            tuple_shapes = '('
            for item in x39:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x39: {}'.format(tuple_shapes))
        else:
            print('x39: {}'.format(x39))
        x40=stochastic_depth(x39, 0.021739130434782608, 'row', False)
        if x40 is None:
            print('x40: {}'.format(x40))
        elif isinstance(x40, torch.Tensor):
            print('x40: {}'.format(x40.shape))
        elif isinstance(x40, tuple):
            tuple_shapes = '('
            for item in x40:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x40: {}'.format(tuple_shapes))
        else:
            print('x40: {}'.format(x40))
        x41=operator.add(x26, x40)
        if x41 is None:
            print('x41: {}'.format(x41))
        elif isinstance(x41, torch.Tensor):
            print('x41: {}'.format(x41.shape))
        elif isinstance(x41, tuple):
            tuple_shapes = '('
            for item in x41:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x41: {}'.format(tuple_shapes))
        else:
            print('x41: {}'.format(x41))
        x42=self.layernorm4(x41)
        if x42 is None:
            print('x42: {}'.format(x42))
        elif isinstance(x42, torch.Tensor):
            print('x42: {}'.format(x42.shape))
        elif isinstance(x42, tuple):
            tuple_shapes = '('
            for item in x42:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x42: {}'.format(tuple_shapes))
        else:
            print('x42: {}'.format(x42))
        x43=self.linear2(x42)
        if x43 is None:
            print('x43: {}'.format(x43))
        elif isinstance(x43, torch.Tensor):
            print('x43: {}'.format(x43.shape))
        elif isinstance(x43, tuple):
            tuple_shapes = '('
            for item in x43:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x43: {}'.format(tuple_shapes))
        else:
            print('x43: {}'.format(x43))
        x44=self.gelu1(x43)
        if x44 is None:
            print('x44: {}'.format(x44))
        elif isinstance(x44, torch.Tensor):
            print('x44: {}'.format(x44.shape))
        elif isinstance(x44, tuple):
            tuple_shapes = '('
            for item in x44:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x44: {}'.format(tuple_shapes))
        else:
            print('x44: {}'.format(x44))
        x45=self.dropout2(x44)
        if x45 is None:
            print('x45: {}'.format(x45))
        elif isinstance(x45, torch.Tensor):
            print('x45: {}'.format(x45.shape))
        elif isinstance(x45, tuple):
            tuple_shapes = '('
            for item in x45:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x45: {}'.format(tuple_shapes))
        else:
            print('x45: {}'.format(x45))
        x46=self.linear3(x45)
        if x46 is None:
            print('x46: {}'.format(x46))
        elif isinstance(x46, torch.Tensor):
            print('x46: {}'.format(x46.shape))
        elif isinstance(x46, tuple):
            tuple_shapes = '('
            for item in x46:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x46: {}'.format(tuple_shapes))
        else:
            print('x46: {}'.format(x46))
        x47=self.dropout3(x46)
        if x47 is None:
            print('x47: {}'.format(x47))
        elif isinstance(x47, torch.Tensor):
            print('x47: {}'.format(x47.shape))
        elif isinstance(x47, tuple):
            tuple_shapes = '('
            for item in x47:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x47: {}'.format(tuple_shapes))
        else:
            print('x47: {}'.format(x47))
        x48=stochastic_depth(x47, 0.021739130434782608, 'row', False)
        if x48 is None:
            print('x48: {}'.format(x48))
        elif isinstance(x48, torch.Tensor):
            print('x48: {}'.format(x48.shape))
        elif isinstance(x48, tuple):
            tuple_shapes = '('
            for item in x48:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x48: {}'.format(tuple_shapes))
        else:
            print('x48: {}'.format(x48))
        x49=operator.add(x41, x48)
        if x49 is None:
            print('x49: {}'.format(x49))
        elif isinstance(x49, torch.Tensor):
            print('x49: {}'.format(x49.shape))
        elif isinstance(x49, tuple):
            tuple_shapes = '('
            for item in x49:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x49: {}'.format(tuple_shapes))
        else:
            print('x49: {}'.format(x49))
        x50=builtins.getattr(x49, 'shape')
        if x50 is None:
            print('x50: {}'.format(x50))
        elif isinstance(x50, torch.Tensor):
            print('x50: {}'.format(x50.shape))
        elif isinstance(x50, tuple):
            tuple_shapes = '('
            for item in x50:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x50: {}'.format(tuple_shapes))
        else:
            print('x50: {}'.format(x50))
        x51=operator.getitem(x50, slice(-3, None, None))
        if x51 is None:
            print('x51: {}'.format(x51))
        elif isinstance(x51, torch.Tensor):
            print('x51: {}'.format(x51.shape))
        elif isinstance(x51, tuple):
            tuple_shapes = '('
            for item in x51:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x51: {}'.format(tuple_shapes))
        else:
            print('x51: {}'.format(x51))
        x52=operator.getitem(x51, 0)
        if x52 is None:
            print('x52: {}'.format(x52))
        elif isinstance(x52, torch.Tensor):
            print('x52: {}'.format(x52.shape))
        elif isinstance(x52, tuple):
            tuple_shapes = '('
            for item in x52:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x52: {}'.format(tuple_shapes))
        else:
            print('x52: {}'.format(x52))
        x53=operator.getitem(x51, 1)
        if x53 is None:
            print('x53: {}'.format(x53))
        elif isinstance(x53, torch.Tensor):
            print('x53: {}'.format(x53.shape))
        elif isinstance(x53, tuple):
            tuple_shapes = '('
            for item in x53:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x53: {}'.format(tuple_shapes))
        else:
            print('x53: {}'.format(x53))
        x54=operator.getitem(x51, 2)
        if x54 is None:
            print('x54: {}'.format(x54))
        elif isinstance(x54, torch.Tensor):
            print('x54: {}'.format(x54.shape))
        elif isinstance(x54, tuple):
            tuple_shapes = '('
            for item in x54:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x54: {}'.format(tuple_shapes))
        else:
            print('x54: {}'.format(x54))
        x55=operator.mod(x53, 2)
        if x55 is None:
            print('x55: {}'.format(x55))
        elif isinstance(x55, torch.Tensor):
            print('x55: {}'.format(x55.shape))
        elif isinstance(x55, tuple):
            tuple_shapes = '('
            for item in x55:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x55: {}'.format(tuple_shapes))
        else:
            print('x55: {}'.format(x55))
        x56=operator.mod(x52, 2)
        if x56 is None:
            print('x56: {}'.format(x56))
        elif isinstance(x56, torch.Tensor):
            print('x56: {}'.format(x56.shape))
        elif isinstance(x56, tuple):
            tuple_shapes = '('
            for item in x56:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x56: {}'.format(tuple_shapes))
        else:
            print('x56: {}'.format(x56))
        x57=torch.nn.functional.pad(x49, (0, 0, 0, x55, 0, x56))
        if x57 is None:
            print('x57: {}'.format(x57))
        elif isinstance(x57, torch.Tensor):
            print('x57: {}'.format(x57.shape))
        elif isinstance(x57, tuple):
            tuple_shapes = '('
            for item in x57:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x57: {}'.format(tuple_shapes))
        else:
            print('x57: {}'.format(x57))
        x58=operator.getitem(x57, (Ellipsis, slice(0, None, 2), slice(0, None, 2), slice(None, None, None)))
        if x58 is None:
            print('x58: {}'.format(x58))
        elif isinstance(x58, torch.Tensor):
            print('x58: {}'.format(x58.shape))
        elif isinstance(x58, tuple):
            tuple_shapes = '('
            for item in x58:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x58: {}'.format(tuple_shapes))
        else:
            print('x58: {}'.format(x58))
        x59=operator.getitem(x57, (Ellipsis, slice(1, None, 2), slice(0, None, 2), slice(None, None, None)))
        if x59 is None:
            print('x59: {}'.format(x59))
        elif isinstance(x59, torch.Tensor):
            print('x59: {}'.format(x59.shape))
        elif isinstance(x59, tuple):
            tuple_shapes = '('
            for item in x59:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x59: {}'.format(tuple_shapes))
        else:
            print('x59: {}'.format(x59))
        x60=operator.getitem(x57, (Ellipsis, slice(0, None, 2), slice(1, None, 2), slice(None, None, None)))
        if x60 is None:
            print('x60: {}'.format(x60))
        elif isinstance(x60, torch.Tensor):
            print('x60: {}'.format(x60.shape))
        elif isinstance(x60, tuple):
            tuple_shapes = '('
            for item in x60:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x60: {}'.format(tuple_shapes))
        else:
            print('x60: {}'.format(x60))
        x61=operator.getitem(x57, (Ellipsis, slice(1, None, 2), slice(1, None, 2), slice(None, None, None)))
        if x61 is None:
            print('x61: {}'.format(x61))
        elif isinstance(x61, torch.Tensor):
            print('x61: {}'.format(x61.shape))
        elif isinstance(x61, tuple):
            tuple_shapes = '('
            for item in x61:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x61: {}'.format(tuple_shapes))
        else:
            print('x61: {}'.format(x61))
        x62=torch.cat([x58, x59, x60, x61], -1)
        if x62 is None:
            print('x62: {}'.format(x62))
        elif isinstance(x62, torch.Tensor):
            print('x62: {}'.format(x62.shape))
        elif isinstance(x62, tuple):
            tuple_shapes = '('
            for item in x62:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x62: {}'.format(tuple_shapes))
        else:
            print('x62: {}'.format(x62))
        x63=self.layernorm5(x62)
        if x63 is None:
            print('x63: {}'.format(x63))
        elif isinstance(x63, torch.Tensor):
            print('x63: {}'.format(x63.shape))
        elif isinstance(x63, tuple):
            tuple_shapes = '('
            for item in x63:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x63: {}'.format(tuple_shapes))
        else:
            print('x63: {}'.format(x63))
        x64=self.linear4(x63)
        if x64 is None:
            print('x64: {}'.format(x64))
        elif isinstance(x64, torch.Tensor):
            print('x64: {}'.format(x64.shape))
        elif isinstance(x64, tuple):
            tuple_shapes = '('
            for item in x64:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x64: {}'.format(tuple_shapes))
        else:
            print('x64: {}'.format(x64))
        x65=self.layernorm6(x64)
        if x65 is None:
            print('x65: {}'.format(x65))
        elif isinstance(x65, torch.Tensor):
            print('x65: {}'.format(x65.shape))
        elif isinstance(x65, tuple):
            tuple_shapes = '('
            for item in x65:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x65: {}'.format(tuple_shapes))
        else:
            print('x65: {}'.format(x65))
        x68=operator.getitem(self.relative_position_bias_table2, self.relative_position_index2)
        if x68 is None:
            print('x68: {}'.format(x68))
        elif isinstance(x68, torch.Tensor):
            print('x68: {}'.format(x68.shape))
        elif isinstance(x68, tuple):
            tuple_shapes = '('
            for item in x68:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x68: {}'.format(tuple_shapes))
        else:
            print('x68: {}'.format(x68))
        x69=x68.view(49, 49, -1)
        if x69 is None:
            print('x69: {}'.format(x69))
        elif isinstance(x69, torch.Tensor):
            print('x69: {}'.format(x69.shape))
        elif isinstance(x69, tuple):
            tuple_shapes = '('
            for item in x69:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x69: {}'.format(tuple_shapes))
        else:
            print('x69: {}'.format(x69))
        x70=x69.permute(2, 0, 1)
        if x70 is None:
            print('x70: {}'.format(x70))
        elif isinstance(x70, torch.Tensor):
            print('x70: {}'.format(x70.shape))
        elif isinstance(x70, tuple):
            tuple_shapes = '('
            for item in x70:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x70: {}'.format(tuple_shapes))
        else:
            print('x70: {}'.format(x70))
        x71=x70.contiguous()
        if x71 is None:
            print('x71: {}'.format(x71))
        elif isinstance(x71, torch.Tensor):
            print('x71: {}'.format(x71.shape))
        elif isinstance(x71, tuple):
            tuple_shapes = '('
            for item in x71:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x71: {}'.format(tuple_shapes))
        else:
            print('x71: {}'.format(x71))
        x72=x71.unsqueeze(0)
        if x72 is None:
            print('x72: {}'.format(x72))
        elif isinstance(x72, torch.Tensor):
            print('x72: {}'.format(x72.shape))
        elif isinstance(x72, tuple):
            tuple_shapes = '('
            for item in x72:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x72: {}'.format(tuple_shapes))
        else:
            print('x72: {}'.format(x72))
        x77=torchvision.models.swin_transformer.shifted_window_attention(x65, self.weight4, self.weight5, x72, [7, 7], 8,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias4, proj_bias=self.bias5)
        if x77 is None:
            print('x77: {}'.format(x77))
        elif isinstance(x77, torch.Tensor):
            print('x77: {}'.format(x77.shape))
        elif isinstance(x77, tuple):
            tuple_shapes = '('
            for item in x77:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x77: {}'.format(tuple_shapes))
        else:
            print('x77: {}'.format(x77))
        x78=stochastic_depth(x77, 0.043478260869565216, 'row', False)
        if x78 is None:
            print('x78: {}'.format(x78))
        elif isinstance(x78, torch.Tensor):
            print('x78: {}'.format(x78.shape))
        elif isinstance(x78, tuple):
            tuple_shapes = '('
            for item in x78:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x78: {}'.format(tuple_shapes))
        else:
            print('x78: {}'.format(x78))
        x79=operator.add(x64, x78)
        if x79 is None:
            print('x79: {}'.format(x79))
        elif isinstance(x79, torch.Tensor):
            print('x79: {}'.format(x79.shape))
        elif isinstance(x79, tuple):
            tuple_shapes = '('
            for item in x79:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x79: {}'.format(tuple_shapes))
        else:
            print('x79: {}'.format(x79))
        x80=self.layernorm7(x79)
        if x80 is None:
            print('x80: {}'.format(x80))
        elif isinstance(x80, torch.Tensor):
            print('x80: {}'.format(x80.shape))
        elif isinstance(x80, tuple):
            tuple_shapes = '('
            for item in x80:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x80: {}'.format(tuple_shapes))
        else:
            print('x80: {}'.format(x80))
        x81=self.linear5(x80)
        if x81 is None:
            print('x81: {}'.format(x81))
        elif isinstance(x81, torch.Tensor):
            print('x81: {}'.format(x81.shape))
        elif isinstance(x81, tuple):
            tuple_shapes = '('
            for item in x81:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x81: {}'.format(tuple_shapes))
        else:
            print('x81: {}'.format(x81))
        x82=self.gelu2(x81)
        if x82 is None:
            print('x82: {}'.format(x82))
        elif isinstance(x82, torch.Tensor):
            print('x82: {}'.format(x82.shape))
        elif isinstance(x82, tuple):
            tuple_shapes = '('
            for item in x82:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x82: {}'.format(tuple_shapes))
        else:
            print('x82: {}'.format(x82))
        x83=self.dropout4(x82)
        if x83 is None:
            print('x83: {}'.format(x83))
        elif isinstance(x83, torch.Tensor):
            print('x83: {}'.format(x83.shape))
        elif isinstance(x83, tuple):
            tuple_shapes = '('
            for item in x83:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x83: {}'.format(tuple_shapes))
        else:
            print('x83: {}'.format(x83))
        x84=self.linear6(x83)
        if x84 is None:
            print('x84: {}'.format(x84))
        elif isinstance(x84, torch.Tensor):
            print('x84: {}'.format(x84.shape))
        elif isinstance(x84, tuple):
            tuple_shapes = '('
            for item in x84:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x84: {}'.format(tuple_shapes))
        else:
            print('x84: {}'.format(x84))
        x85=self.dropout5(x84)
        if x85 is None:
            print('x85: {}'.format(x85))
        elif isinstance(x85, torch.Tensor):
            print('x85: {}'.format(x85.shape))
        elif isinstance(x85, tuple):
            tuple_shapes = '('
            for item in x85:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x85: {}'.format(tuple_shapes))
        else:
            print('x85: {}'.format(x85))
        x86=stochastic_depth(x85, 0.043478260869565216, 'row', False)
        if x86 is None:
            print('x86: {}'.format(x86))
        elif isinstance(x86, torch.Tensor):
            print('x86: {}'.format(x86.shape))
        elif isinstance(x86, tuple):
            tuple_shapes = '('
            for item in x86:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x86: {}'.format(tuple_shapes))
        else:
            print('x86: {}'.format(x86))
        x87=operator.add(x79, x86)
        if x87 is None:
            print('x87: {}'.format(x87))
        elif isinstance(x87, torch.Tensor):
            print('x87: {}'.format(x87.shape))
        elif isinstance(x87, tuple):
            tuple_shapes = '('
            for item in x87:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x87: {}'.format(tuple_shapes))
        else:
            print('x87: {}'.format(x87))
        x88=self.layernorm8(x87)
        if x88 is None:
            print('x88: {}'.format(x88))
        elif isinstance(x88, torch.Tensor):
            print('x88: {}'.format(x88.shape))
        elif isinstance(x88, tuple):
            tuple_shapes = '('
            for item in x88:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x88: {}'.format(tuple_shapes))
        else:
            print('x88: {}'.format(x88))
        x91=operator.getitem(self.relative_position_bias_table3, self.relative_position_index3)
        if x91 is None:
            print('x91: {}'.format(x91))
        elif isinstance(x91, torch.Tensor):
            print('x91: {}'.format(x91.shape))
        elif isinstance(x91, tuple):
            tuple_shapes = '('
            for item in x91:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x91: {}'.format(tuple_shapes))
        else:
            print('x91: {}'.format(x91))
        x92=x91.view(49, 49, -1)
        if x92 is None:
            print('x92: {}'.format(x92))
        elif isinstance(x92, torch.Tensor):
            print('x92: {}'.format(x92.shape))
        elif isinstance(x92, tuple):
            tuple_shapes = '('
            for item in x92:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x92: {}'.format(tuple_shapes))
        else:
            print('x92: {}'.format(x92))
        x93=x92.permute(2, 0, 1)
        if x93 is None:
            print('x93: {}'.format(x93))
        elif isinstance(x93, torch.Tensor):
            print('x93: {}'.format(x93.shape))
        elif isinstance(x93, tuple):
            tuple_shapes = '('
            for item in x93:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x93: {}'.format(tuple_shapes))
        else:
            print('x93: {}'.format(x93))
        x94=x93.contiguous()
        if x94 is None:
            print('x94: {}'.format(x94))
        elif isinstance(x94, torch.Tensor):
            print('x94: {}'.format(x94.shape))
        elif isinstance(x94, tuple):
            tuple_shapes = '('
            for item in x94:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x94: {}'.format(tuple_shapes))
        else:
            print('x94: {}'.format(x94))
        x95=x94.unsqueeze(0)
        if x95 is None:
            print('x95: {}'.format(x95))
        elif isinstance(x95, torch.Tensor):
            print('x95: {}'.format(x95.shape))
        elif isinstance(x95, tuple):
            tuple_shapes = '('
            for item in x95:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x95: {}'.format(tuple_shapes))
        else:
            print('x95: {}'.format(x95))
        x100=torchvision.models.swin_transformer.shifted_window_attention(x88, self.weight6, self.weight7, x95, [7, 7], 8,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias6, proj_bias=self.bias7)
        if x100 is None:
            print('x100: {}'.format(x100))
        elif isinstance(x100, torch.Tensor):
            print('x100: {}'.format(x100.shape))
        elif isinstance(x100, tuple):
            tuple_shapes = '('
            for item in x100:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x100: {}'.format(tuple_shapes))
        else:
            print('x100: {}'.format(x100))
        x101=stochastic_depth(x100, 0.06521739130434782, 'row', False)
        if x101 is None:
            print('x101: {}'.format(x101))
        elif isinstance(x101, torch.Tensor):
            print('x101: {}'.format(x101.shape))
        elif isinstance(x101, tuple):
            tuple_shapes = '('
            for item in x101:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x101: {}'.format(tuple_shapes))
        else:
            print('x101: {}'.format(x101))
        x102=operator.add(x87, x101)
        if x102 is None:
            print('x102: {}'.format(x102))
        elif isinstance(x102, torch.Tensor):
            print('x102: {}'.format(x102.shape))
        elif isinstance(x102, tuple):
            tuple_shapes = '('
            for item in x102:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x102: {}'.format(tuple_shapes))
        else:
            print('x102: {}'.format(x102))
        x103=self.layernorm9(x102)
        if x103 is None:
            print('x103: {}'.format(x103))
        elif isinstance(x103, torch.Tensor):
            print('x103: {}'.format(x103.shape))
        elif isinstance(x103, tuple):
            tuple_shapes = '('
            for item in x103:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x103: {}'.format(tuple_shapes))
        else:
            print('x103: {}'.format(x103))
        x104=self.linear7(x103)
        if x104 is None:
            print('x104: {}'.format(x104))
        elif isinstance(x104, torch.Tensor):
            print('x104: {}'.format(x104.shape))
        elif isinstance(x104, tuple):
            tuple_shapes = '('
            for item in x104:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x104: {}'.format(tuple_shapes))
        else:
            print('x104: {}'.format(x104))
        x105=self.gelu3(x104)
        if x105 is None:
            print('x105: {}'.format(x105))
        elif isinstance(x105, torch.Tensor):
            print('x105: {}'.format(x105.shape))
        elif isinstance(x105, tuple):
            tuple_shapes = '('
            for item in x105:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x105: {}'.format(tuple_shapes))
        else:
            print('x105: {}'.format(x105))
        x106=self.dropout6(x105)
        if x106 is None:
            print('x106: {}'.format(x106))
        elif isinstance(x106, torch.Tensor):
            print('x106: {}'.format(x106.shape))
        elif isinstance(x106, tuple):
            tuple_shapes = '('
            for item in x106:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x106: {}'.format(tuple_shapes))
        else:
            print('x106: {}'.format(x106))
        x107=self.linear8(x106)
        if x107 is None:
            print('x107: {}'.format(x107))
        elif isinstance(x107, torch.Tensor):
            print('x107: {}'.format(x107.shape))
        elif isinstance(x107, tuple):
            tuple_shapes = '('
            for item in x107:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x107: {}'.format(tuple_shapes))
        else:
            print('x107: {}'.format(x107))
        x108=self.dropout7(x107)
        if x108 is None:
            print('x108: {}'.format(x108))
        elif isinstance(x108, torch.Tensor):
            print('x108: {}'.format(x108.shape))
        elif isinstance(x108, tuple):
            tuple_shapes = '('
            for item in x108:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x108: {}'.format(tuple_shapes))
        else:
            print('x108: {}'.format(x108))
        x109=stochastic_depth(x108, 0.06521739130434782, 'row', False)
        if x109 is None:
            print('x109: {}'.format(x109))
        elif isinstance(x109, torch.Tensor):
            print('x109: {}'.format(x109.shape))
        elif isinstance(x109, tuple):
            tuple_shapes = '('
            for item in x109:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x109: {}'.format(tuple_shapes))
        else:
            print('x109: {}'.format(x109))
        x110=operator.add(x102, x109)
        if x110 is None:
            print('x110: {}'.format(x110))
        elif isinstance(x110, torch.Tensor):
            print('x110: {}'.format(x110.shape))
        elif isinstance(x110, tuple):
            tuple_shapes = '('
            for item in x110:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x110: {}'.format(tuple_shapes))
        else:
            print('x110: {}'.format(x110))
        x111=builtins.getattr(x110, 'shape')
        if x111 is None:
            print('x111: {}'.format(x111))
        elif isinstance(x111, torch.Tensor):
            print('x111: {}'.format(x111.shape))
        elif isinstance(x111, tuple):
            tuple_shapes = '('
            for item in x111:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x111: {}'.format(tuple_shapes))
        else:
            print('x111: {}'.format(x111))
        x112=operator.getitem(x111, slice(-3, None, None))
        if x112 is None:
            print('x112: {}'.format(x112))
        elif isinstance(x112, torch.Tensor):
            print('x112: {}'.format(x112.shape))
        elif isinstance(x112, tuple):
            tuple_shapes = '('
            for item in x112:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x112: {}'.format(tuple_shapes))
        else:
            print('x112: {}'.format(x112))
        x113=operator.getitem(x112, 0)
        if x113 is None:
            print('x113: {}'.format(x113))
        elif isinstance(x113, torch.Tensor):
            print('x113: {}'.format(x113.shape))
        elif isinstance(x113, tuple):
            tuple_shapes = '('
            for item in x113:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x113: {}'.format(tuple_shapes))
        else:
            print('x113: {}'.format(x113))
        x114=operator.getitem(x112, 1)
        if x114 is None:
            print('x114: {}'.format(x114))
        elif isinstance(x114, torch.Tensor):
            print('x114: {}'.format(x114.shape))
        elif isinstance(x114, tuple):
            tuple_shapes = '('
            for item in x114:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x114: {}'.format(tuple_shapes))
        else:
            print('x114: {}'.format(x114))
        x115=operator.getitem(x112, 2)
        if x115 is None:
            print('x115: {}'.format(x115))
        elif isinstance(x115, torch.Tensor):
            print('x115: {}'.format(x115.shape))
        elif isinstance(x115, tuple):
            tuple_shapes = '('
            for item in x115:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x115: {}'.format(tuple_shapes))
        else:
            print('x115: {}'.format(x115))
        x116=operator.mod(x114, 2)
        if x116 is None:
            print('x116: {}'.format(x116))
        elif isinstance(x116, torch.Tensor):
            print('x116: {}'.format(x116.shape))
        elif isinstance(x116, tuple):
            tuple_shapes = '('
            for item in x116:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x116: {}'.format(tuple_shapes))
        else:
            print('x116: {}'.format(x116))
        x117=operator.mod(x113, 2)
        if x117 is None:
            print('x117: {}'.format(x117))
        elif isinstance(x117, torch.Tensor):
            print('x117: {}'.format(x117.shape))
        elif isinstance(x117, tuple):
            tuple_shapes = '('
            for item in x117:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x117: {}'.format(tuple_shapes))
        else:
            print('x117: {}'.format(x117))
        x118=torch.nn.functional.pad(x110, (0, 0, 0, x116, 0, x117))
        if x118 is None:
            print('x118: {}'.format(x118))
        elif isinstance(x118, torch.Tensor):
            print('x118: {}'.format(x118.shape))
        elif isinstance(x118, tuple):
            tuple_shapes = '('
            for item in x118:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x118: {}'.format(tuple_shapes))
        else:
            print('x118: {}'.format(x118))
        x119=operator.getitem(x118, (Ellipsis, slice(0, None, 2), slice(0, None, 2), slice(None, None, None)))
        if x119 is None:
            print('x119: {}'.format(x119))
        elif isinstance(x119, torch.Tensor):
            print('x119: {}'.format(x119.shape))
        elif isinstance(x119, tuple):
            tuple_shapes = '('
            for item in x119:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x119: {}'.format(tuple_shapes))
        else:
            print('x119: {}'.format(x119))
        x120=operator.getitem(x118, (Ellipsis, slice(1, None, 2), slice(0, None, 2), slice(None, None, None)))
        if x120 is None:
            print('x120: {}'.format(x120))
        elif isinstance(x120, torch.Tensor):
            print('x120: {}'.format(x120.shape))
        elif isinstance(x120, tuple):
            tuple_shapes = '('
            for item in x120:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x120: {}'.format(tuple_shapes))
        else:
            print('x120: {}'.format(x120))
        x121=operator.getitem(x118, (Ellipsis, slice(0, None, 2), slice(1, None, 2), slice(None, None, None)))
        if x121 is None:
            print('x121: {}'.format(x121))
        elif isinstance(x121, torch.Tensor):
            print('x121: {}'.format(x121.shape))
        elif isinstance(x121, tuple):
            tuple_shapes = '('
            for item in x121:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x121: {}'.format(tuple_shapes))
        else:
            print('x121: {}'.format(x121))
        x122=operator.getitem(x118, (Ellipsis, slice(1, None, 2), slice(1, None, 2), slice(None, None, None)))
        if x122 is None:
            print('x122: {}'.format(x122))
        elif isinstance(x122, torch.Tensor):
            print('x122: {}'.format(x122.shape))
        elif isinstance(x122, tuple):
            tuple_shapes = '('
            for item in x122:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x122: {}'.format(tuple_shapes))
        else:
            print('x122: {}'.format(x122))
        x123=torch.cat([x119, x120, x121, x122], -1)
        if x123 is None:
            print('x123: {}'.format(x123))
        elif isinstance(x123, torch.Tensor):
            print('x123: {}'.format(x123.shape))
        elif isinstance(x123, tuple):
            tuple_shapes = '('
            for item in x123:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x123: {}'.format(tuple_shapes))
        else:
            print('x123: {}'.format(x123))
        x124=self.layernorm10(x123)
        if x124 is None:
            print('x124: {}'.format(x124))
        elif isinstance(x124, torch.Tensor):
            print('x124: {}'.format(x124.shape))
        elif isinstance(x124, tuple):
            tuple_shapes = '('
            for item in x124:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x124: {}'.format(tuple_shapes))
        else:
            print('x124: {}'.format(x124))
        x125=self.linear9(x124)
        if x125 is None:
            print('x125: {}'.format(x125))
        elif isinstance(x125, torch.Tensor):
            print('x125: {}'.format(x125.shape))
        elif isinstance(x125, tuple):
            tuple_shapes = '('
            for item in x125:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x125: {}'.format(tuple_shapes))
        else:
            print('x125: {}'.format(x125))
        x126=self.layernorm11(x125)
        if x126 is None:
            print('x126: {}'.format(x126))
        elif isinstance(x126, torch.Tensor):
            print('x126: {}'.format(x126.shape))
        elif isinstance(x126, tuple):
            tuple_shapes = '('
            for item in x126:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x126: {}'.format(tuple_shapes))
        else:
            print('x126: {}'.format(x126))
        x129=operator.getitem(self.relative_position_bias_table4, self.relative_position_index4)
        if x129 is None:
            print('x129: {}'.format(x129))
        elif isinstance(x129, torch.Tensor):
            print('x129: {}'.format(x129.shape))
        elif isinstance(x129, tuple):
            tuple_shapes = '('
            for item in x129:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x129: {}'.format(tuple_shapes))
        else:
            print('x129: {}'.format(x129))
        x130=x129.view(49, 49, -1)
        if x130 is None:
            print('x130: {}'.format(x130))
        elif isinstance(x130, torch.Tensor):
            print('x130: {}'.format(x130.shape))
        elif isinstance(x130, tuple):
            tuple_shapes = '('
            for item in x130:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x130: {}'.format(tuple_shapes))
        else:
            print('x130: {}'.format(x130))
        x131=x130.permute(2, 0, 1)
        if x131 is None:
            print('x131: {}'.format(x131))
        elif isinstance(x131, torch.Tensor):
            print('x131: {}'.format(x131.shape))
        elif isinstance(x131, tuple):
            tuple_shapes = '('
            for item in x131:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x131: {}'.format(tuple_shapes))
        else:
            print('x131: {}'.format(x131))
        x132=x131.contiguous()
        if x132 is None:
            print('x132: {}'.format(x132))
        elif isinstance(x132, torch.Tensor):
            print('x132: {}'.format(x132.shape))
        elif isinstance(x132, tuple):
            tuple_shapes = '('
            for item in x132:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x132: {}'.format(tuple_shapes))
        else:
            print('x132: {}'.format(x132))
        x133=x132.unsqueeze(0)
        if x133 is None:
            print('x133: {}'.format(x133))
        elif isinstance(x133, torch.Tensor):
            print('x133: {}'.format(x133.shape))
        elif isinstance(x133, tuple):
            tuple_shapes = '('
            for item in x133:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x133: {}'.format(tuple_shapes))
        else:
            print('x133: {}'.format(x133))
        x138=torchvision.models.swin_transformer.shifted_window_attention(x126, self.weight8, self.weight9, x133, [7, 7], 16,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias8, proj_bias=self.bias9)
        if x138 is None:
            print('x138: {}'.format(x138))
        elif isinstance(x138, torch.Tensor):
            print('x138: {}'.format(x138.shape))
        elif isinstance(x138, tuple):
            tuple_shapes = '('
            for item in x138:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x138: {}'.format(tuple_shapes))
        else:
            print('x138: {}'.format(x138))
        x139=stochastic_depth(x138, 0.08695652173913043, 'row', False)
        if x139 is None:
            print('x139: {}'.format(x139))
        elif isinstance(x139, torch.Tensor):
            print('x139: {}'.format(x139.shape))
        elif isinstance(x139, tuple):
            tuple_shapes = '('
            for item in x139:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x139: {}'.format(tuple_shapes))
        else:
            print('x139: {}'.format(x139))
        x140=operator.add(x125, x139)
        if x140 is None:
            print('x140: {}'.format(x140))
        elif isinstance(x140, torch.Tensor):
            print('x140: {}'.format(x140.shape))
        elif isinstance(x140, tuple):
            tuple_shapes = '('
            for item in x140:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x140: {}'.format(tuple_shapes))
        else:
            print('x140: {}'.format(x140))
        x141=self.layernorm12(x140)
        if x141 is None:
            print('x141: {}'.format(x141))
        elif isinstance(x141, torch.Tensor):
            print('x141: {}'.format(x141.shape))
        elif isinstance(x141, tuple):
            tuple_shapes = '('
            for item in x141:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x141: {}'.format(tuple_shapes))
        else:
            print('x141: {}'.format(x141))
        x142=self.linear10(x141)
        if x142 is None:
            print('x142: {}'.format(x142))
        elif isinstance(x142, torch.Tensor):
            print('x142: {}'.format(x142.shape))
        elif isinstance(x142, tuple):
            tuple_shapes = '('
            for item in x142:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x142: {}'.format(tuple_shapes))
        else:
            print('x142: {}'.format(x142))
        x143=self.gelu4(x142)
        if x143 is None:
            print('x143: {}'.format(x143))
        elif isinstance(x143, torch.Tensor):
            print('x143: {}'.format(x143.shape))
        elif isinstance(x143, tuple):
            tuple_shapes = '('
            for item in x143:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x143: {}'.format(tuple_shapes))
        else:
            print('x143: {}'.format(x143))
        x144=self.dropout8(x143)
        if x144 is None:
            print('x144: {}'.format(x144))
        elif isinstance(x144, torch.Tensor):
            print('x144: {}'.format(x144.shape))
        elif isinstance(x144, tuple):
            tuple_shapes = '('
            for item in x144:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x144: {}'.format(tuple_shapes))
        else:
            print('x144: {}'.format(x144))
        x145=self.linear11(x144)
        if x145 is None:
            print('x145: {}'.format(x145))
        elif isinstance(x145, torch.Tensor):
            print('x145: {}'.format(x145.shape))
        elif isinstance(x145, tuple):
            tuple_shapes = '('
            for item in x145:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x145: {}'.format(tuple_shapes))
        else:
            print('x145: {}'.format(x145))
        x146=self.dropout9(x145)
        if x146 is None:
            print('x146: {}'.format(x146))
        elif isinstance(x146, torch.Tensor):
            print('x146: {}'.format(x146.shape))
        elif isinstance(x146, tuple):
            tuple_shapes = '('
            for item in x146:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x146: {}'.format(tuple_shapes))
        else:
            print('x146: {}'.format(x146))
        x147=stochastic_depth(x146, 0.08695652173913043, 'row', False)
        if x147 is None:
            print('x147: {}'.format(x147))
        elif isinstance(x147, torch.Tensor):
            print('x147: {}'.format(x147.shape))
        elif isinstance(x147, tuple):
            tuple_shapes = '('
            for item in x147:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x147: {}'.format(tuple_shapes))
        else:
            print('x147: {}'.format(x147))
        x148=operator.add(x140, x147)
        if x148 is None:
            print('x148: {}'.format(x148))
        elif isinstance(x148, torch.Tensor):
            print('x148: {}'.format(x148.shape))
        elif isinstance(x148, tuple):
            tuple_shapes = '('
            for item in x148:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x148: {}'.format(tuple_shapes))
        else:
            print('x148: {}'.format(x148))
        x149=self.layernorm13(x148)
        if x149 is None:
            print('x149: {}'.format(x149))
        elif isinstance(x149, torch.Tensor):
            print('x149: {}'.format(x149.shape))
        elif isinstance(x149, tuple):
            tuple_shapes = '('
            for item in x149:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x149: {}'.format(tuple_shapes))
        else:
            print('x149: {}'.format(x149))
        x152=operator.getitem(self.relative_position_bias_table5, self.relative_position_index5)
        if x152 is None:
            print('x152: {}'.format(x152))
        elif isinstance(x152, torch.Tensor):
            print('x152: {}'.format(x152.shape))
        elif isinstance(x152, tuple):
            tuple_shapes = '('
            for item in x152:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x152: {}'.format(tuple_shapes))
        else:
            print('x152: {}'.format(x152))
        x153=x152.view(49, 49, -1)
        if x153 is None:
            print('x153: {}'.format(x153))
        elif isinstance(x153, torch.Tensor):
            print('x153: {}'.format(x153.shape))
        elif isinstance(x153, tuple):
            tuple_shapes = '('
            for item in x153:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x153: {}'.format(tuple_shapes))
        else:
            print('x153: {}'.format(x153))
        x154=x153.permute(2, 0, 1)
        if x154 is None:
            print('x154: {}'.format(x154))
        elif isinstance(x154, torch.Tensor):
            print('x154: {}'.format(x154.shape))
        elif isinstance(x154, tuple):
            tuple_shapes = '('
            for item in x154:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x154: {}'.format(tuple_shapes))
        else:
            print('x154: {}'.format(x154))
        x155=x154.contiguous()
        if x155 is None:
            print('x155: {}'.format(x155))
        elif isinstance(x155, torch.Tensor):
            print('x155: {}'.format(x155.shape))
        elif isinstance(x155, tuple):
            tuple_shapes = '('
            for item in x155:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x155: {}'.format(tuple_shapes))
        else:
            print('x155: {}'.format(x155))
        x156=x155.unsqueeze(0)
        if x156 is None:
            print('x156: {}'.format(x156))
        elif isinstance(x156, torch.Tensor):
            print('x156: {}'.format(x156.shape))
        elif isinstance(x156, tuple):
            tuple_shapes = '('
            for item in x156:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x156: {}'.format(tuple_shapes))
        else:
            print('x156: {}'.format(x156))
        x161=torchvision.models.swin_transformer.shifted_window_attention(x149, self.weight10, self.weight11, x156, [7, 7], 16,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias10, proj_bias=self.bias11)
        if x161 is None:
            print('x161: {}'.format(x161))
        elif isinstance(x161, torch.Tensor):
            print('x161: {}'.format(x161.shape))
        elif isinstance(x161, tuple):
            tuple_shapes = '('
            for item in x161:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x161: {}'.format(tuple_shapes))
        else:
            print('x161: {}'.format(x161))
        x162=stochastic_depth(x161, 0.10869565217391304, 'row', False)
        if x162 is None:
            print('x162: {}'.format(x162))
        elif isinstance(x162, torch.Tensor):
            print('x162: {}'.format(x162.shape))
        elif isinstance(x162, tuple):
            tuple_shapes = '('
            for item in x162:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x162: {}'.format(tuple_shapes))
        else:
            print('x162: {}'.format(x162))
        x163=operator.add(x148, x162)
        if x163 is None:
            print('x163: {}'.format(x163))
        elif isinstance(x163, torch.Tensor):
            print('x163: {}'.format(x163.shape))
        elif isinstance(x163, tuple):
            tuple_shapes = '('
            for item in x163:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x163: {}'.format(tuple_shapes))
        else:
            print('x163: {}'.format(x163))
        x164=self.layernorm14(x163)
        if x164 is None:
            print('x164: {}'.format(x164))
        elif isinstance(x164, torch.Tensor):
            print('x164: {}'.format(x164.shape))
        elif isinstance(x164, tuple):
            tuple_shapes = '('
            for item in x164:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x164: {}'.format(tuple_shapes))
        else:
            print('x164: {}'.format(x164))
        x165=self.linear12(x164)
        if x165 is None:
            print('x165: {}'.format(x165))
        elif isinstance(x165, torch.Tensor):
            print('x165: {}'.format(x165.shape))
        elif isinstance(x165, tuple):
            tuple_shapes = '('
            for item in x165:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x165: {}'.format(tuple_shapes))
        else:
            print('x165: {}'.format(x165))
        x166=self.gelu5(x165)
        if x166 is None:
            print('x166: {}'.format(x166))
        elif isinstance(x166, torch.Tensor):
            print('x166: {}'.format(x166.shape))
        elif isinstance(x166, tuple):
            tuple_shapes = '('
            for item in x166:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x166: {}'.format(tuple_shapes))
        else:
            print('x166: {}'.format(x166))
        x167=self.dropout10(x166)
        if x167 is None:
            print('x167: {}'.format(x167))
        elif isinstance(x167, torch.Tensor):
            print('x167: {}'.format(x167.shape))
        elif isinstance(x167, tuple):
            tuple_shapes = '('
            for item in x167:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x167: {}'.format(tuple_shapes))
        else:
            print('x167: {}'.format(x167))
        x168=self.linear13(x167)
        if x168 is None:
            print('x168: {}'.format(x168))
        elif isinstance(x168, torch.Tensor):
            print('x168: {}'.format(x168.shape))
        elif isinstance(x168, tuple):
            tuple_shapes = '('
            for item in x168:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x168: {}'.format(tuple_shapes))
        else:
            print('x168: {}'.format(x168))
        x169=self.dropout11(x168)
        if x169 is None:
            print('x169: {}'.format(x169))
        elif isinstance(x169, torch.Tensor):
            print('x169: {}'.format(x169.shape))
        elif isinstance(x169, tuple):
            tuple_shapes = '('
            for item in x169:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x169: {}'.format(tuple_shapes))
        else:
            print('x169: {}'.format(x169))
        x170=stochastic_depth(x169, 0.10869565217391304, 'row', False)
        if x170 is None:
            print('x170: {}'.format(x170))
        elif isinstance(x170, torch.Tensor):
            print('x170: {}'.format(x170.shape))
        elif isinstance(x170, tuple):
            tuple_shapes = '('
            for item in x170:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x170: {}'.format(tuple_shapes))
        else:
            print('x170: {}'.format(x170))
        x171=operator.add(x163, x170)
        if x171 is None:
            print('x171: {}'.format(x171))
        elif isinstance(x171, torch.Tensor):
            print('x171: {}'.format(x171.shape))
        elif isinstance(x171, tuple):
            tuple_shapes = '('
            for item in x171:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x171: {}'.format(tuple_shapes))
        else:
            print('x171: {}'.format(x171))
        x172=self.layernorm15(x171)
        if x172 is None:
            print('x172: {}'.format(x172))
        elif isinstance(x172, torch.Tensor):
            print('x172: {}'.format(x172.shape))
        elif isinstance(x172, tuple):
            tuple_shapes = '('
            for item in x172:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x172: {}'.format(tuple_shapes))
        else:
            print('x172: {}'.format(x172))
        x175=operator.getitem(self.relative_position_bias_table6, self.relative_position_index6)
        if x175 is None:
            print('x175: {}'.format(x175))
        elif isinstance(x175, torch.Tensor):
            print('x175: {}'.format(x175.shape))
        elif isinstance(x175, tuple):
            tuple_shapes = '('
            for item in x175:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x175: {}'.format(tuple_shapes))
        else:
            print('x175: {}'.format(x175))
        x176=x175.view(49, 49, -1)
        if x176 is None:
            print('x176: {}'.format(x176))
        elif isinstance(x176, torch.Tensor):
            print('x176: {}'.format(x176.shape))
        elif isinstance(x176, tuple):
            tuple_shapes = '('
            for item in x176:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x176: {}'.format(tuple_shapes))
        else:
            print('x176: {}'.format(x176))
        x177=x176.permute(2, 0, 1)
        if x177 is None:
            print('x177: {}'.format(x177))
        elif isinstance(x177, torch.Tensor):
            print('x177: {}'.format(x177.shape))
        elif isinstance(x177, tuple):
            tuple_shapes = '('
            for item in x177:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x177: {}'.format(tuple_shapes))
        else:
            print('x177: {}'.format(x177))
        x178=x177.contiguous()
        if x178 is None:
            print('x178: {}'.format(x178))
        elif isinstance(x178, torch.Tensor):
            print('x178: {}'.format(x178.shape))
        elif isinstance(x178, tuple):
            tuple_shapes = '('
            for item in x178:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x178: {}'.format(tuple_shapes))
        else:
            print('x178: {}'.format(x178))
        x179=x178.unsqueeze(0)
        if x179 is None:
            print('x179: {}'.format(x179))
        elif isinstance(x179, torch.Tensor):
            print('x179: {}'.format(x179.shape))
        elif isinstance(x179, tuple):
            tuple_shapes = '('
            for item in x179:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x179: {}'.format(tuple_shapes))
        else:
            print('x179: {}'.format(x179))
        x184=torchvision.models.swin_transformer.shifted_window_attention(x172, self.weight12, self.weight13, x179, [7, 7], 16,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias12, proj_bias=self.bias13)
        if x184 is None:
            print('x184: {}'.format(x184))
        elif isinstance(x184, torch.Tensor):
            print('x184: {}'.format(x184.shape))
        elif isinstance(x184, tuple):
            tuple_shapes = '('
            for item in x184:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x184: {}'.format(tuple_shapes))
        else:
            print('x184: {}'.format(x184))
        x185=stochastic_depth(x184, 0.13043478260869565, 'row', False)
        if x185 is None:
            print('x185: {}'.format(x185))
        elif isinstance(x185, torch.Tensor):
            print('x185: {}'.format(x185.shape))
        elif isinstance(x185, tuple):
            tuple_shapes = '('
            for item in x185:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x185: {}'.format(tuple_shapes))
        else:
            print('x185: {}'.format(x185))
        x186=operator.add(x171, x185)
        if x186 is None:
            print('x186: {}'.format(x186))
        elif isinstance(x186, torch.Tensor):
            print('x186: {}'.format(x186.shape))
        elif isinstance(x186, tuple):
            tuple_shapes = '('
            for item in x186:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x186: {}'.format(tuple_shapes))
        else:
            print('x186: {}'.format(x186))
        x187=self.layernorm16(x186)
        if x187 is None:
            print('x187: {}'.format(x187))
        elif isinstance(x187, torch.Tensor):
            print('x187: {}'.format(x187.shape))
        elif isinstance(x187, tuple):
            tuple_shapes = '('
            for item in x187:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x187: {}'.format(tuple_shapes))
        else:
            print('x187: {}'.format(x187))
        x188=self.linear14(x187)
        if x188 is None:
            print('x188: {}'.format(x188))
        elif isinstance(x188, torch.Tensor):
            print('x188: {}'.format(x188.shape))
        elif isinstance(x188, tuple):
            tuple_shapes = '('
            for item in x188:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x188: {}'.format(tuple_shapes))
        else:
            print('x188: {}'.format(x188))
        x189=self.gelu6(x188)
        if x189 is None:
            print('x189: {}'.format(x189))
        elif isinstance(x189, torch.Tensor):
            print('x189: {}'.format(x189.shape))
        elif isinstance(x189, tuple):
            tuple_shapes = '('
            for item in x189:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x189: {}'.format(tuple_shapes))
        else:
            print('x189: {}'.format(x189))
        x190=self.dropout12(x189)
        if x190 is None:
            print('x190: {}'.format(x190))
        elif isinstance(x190, torch.Tensor):
            print('x190: {}'.format(x190.shape))
        elif isinstance(x190, tuple):
            tuple_shapes = '('
            for item in x190:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x190: {}'.format(tuple_shapes))
        else:
            print('x190: {}'.format(x190))
        x191=self.linear15(x190)
        if x191 is None:
            print('x191: {}'.format(x191))
        elif isinstance(x191, torch.Tensor):
            print('x191: {}'.format(x191.shape))
        elif isinstance(x191, tuple):
            tuple_shapes = '('
            for item in x191:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x191: {}'.format(tuple_shapes))
        else:
            print('x191: {}'.format(x191))
        x192=self.dropout13(x191)
        if x192 is None:
            print('x192: {}'.format(x192))
        elif isinstance(x192, torch.Tensor):
            print('x192: {}'.format(x192.shape))
        elif isinstance(x192, tuple):
            tuple_shapes = '('
            for item in x192:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x192: {}'.format(tuple_shapes))
        else:
            print('x192: {}'.format(x192))
        x193=stochastic_depth(x192, 0.13043478260869565, 'row', False)
        if x193 is None:
            print('x193: {}'.format(x193))
        elif isinstance(x193, torch.Tensor):
            print('x193: {}'.format(x193.shape))
        elif isinstance(x193, tuple):
            tuple_shapes = '('
            for item in x193:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x193: {}'.format(tuple_shapes))
        else:
            print('x193: {}'.format(x193))
        x194=operator.add(x186, x193)
        if x194 is None:
            print('x194: {}'.format(x194))
        elif isinstance(x194, torch.Tensor):
            print('x194: {}'.format(x194.shape))
        elif isinstance(x194, tuple):
            tuple_shapes = '('
            for item in x194:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x194: {}'.format(tuple_shapes))
        else:
            print('x194: {}'.format(x194))
        x195=self.layernorm17(x194)
        if x195 is None:
            print('x195: {}'.format(x195))
        elif isinstance(x195, torch.Tensor):
            print('x195: {}'.format(x195.shape))
        elif isinstance(x195, tuple):
            tuple_shapes = '('
            for item in x195:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x195: {}'.format(tuple_shapes))
        else:
            print('x195: {}'.format(x195))
        x198=operator.getitem(self.relative_position_bias_table7, self.relative_position_index7)
        if x198 is None:
            print('x198: {}'.format(x198))
        elif isinstance(x198, torch.Tensor):
            print('x198: {}'.format(x198.shape))
        elif isinstance(x198, tuple):
            tuple_shapes = '('
            for item in x198:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x198: {}'.format(tuple_shapes))
        else:
            print('x198: {}'.format(x198))
        x199=x198.view(49, 49, -1)
        if x199 is None:
            print('x199: {}'.format(x199))
        elif isinstance(x199, torch.Tensor):
            print('x199: {}'.format(x199.shape))
        elif isinstance(x199, tuple):
            tuple_shapes = '('
            for item in x199:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x199: {}'.format(tuple_shapes))
        else:
            print('x199: {}'.format(x199))
        x200=x199.permute(2, 0, 1)
        if x200 is None:
            print('x200: {}'.format(x200))
        elif isinstance(x200, torch.Tensor):
            print('x200: {}'.format(x200.shape))
        elif isinstance(x200, tuple):
            tuple_shapes = '('
            for item in x200:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x200: {}'.format(tuple_shapes))
        else:
            print('x200: {}'.format(x200))
        x201=x200.contiguous()
        if x201 is None:
            print('x201: {}'.format(x201))
        elif isinstance(x201, torch.Tensor):
            print('x201: {}'.format(x201.shape))
        elif isinstance(x201, tuple):
            tuple_shapes = '('
            for item in x201:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x201: {}'.format(tuple_shapes))
        else:
            print('x201: {}'.format(x201))
        x202=x201.unsqueeze(0)
        if x202 is None:
            print('x202: {}'.format(x202))
        elif isinstance(x202, torch.Tensor):
            print('x202: {}'.format(x202.shape))
        elif isinstance(x202, tuple):
            tuple_shapes = '('
            for item in x202:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x202: {}'.format(tuple_shapes))
        else:
            print('x202: {}'.format(x202))
        x207=torchvision.models.swin_transformer.shifted_window_attention(x195, self.weight14, self.weight15, x202, [7, 7], 16,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias14, proj_bias=self.bias15)
        if x207 is None:
            print('x207: {}'.format(x207))
        elif isinstance(x207, torch.Tensor):
            print('x207: {}'.format(x207.shape))
        elif isinstance(x207, tuple):
            tuple_shapes = '('
            for item in x207:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x207: {}'.format(tuple_shapes))
        else:
            print('x207: {}'.format(x207))
        x208=stochastic_depth(x207, 0.15217391304347827, 'row', False)
        if x208 is None:
            print('x208: {}'.format(x208))
        elif isinstance(x208, torch.Tensor):
            print('x208: {}'.format(x208.shape))
        elif isinstance(x208, tuple):
            tuple_shapes = '('
            for item in x208:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x208: {}'.format(tuple_shapes))
        else:
            print('x208: {}'.format(x208))
        x209=operator.add(x194, x208)
        if x209 is None:
            print('x209: {}'.format(x209))
        elif isinstance(x209, torch.Tensor):
            print('x209: {}'.format(x209.shape))
        elif isinstance(x209, tuple):
            tuple_shapes = '('
            for item in x209:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x209: {}'.format(tuple_shapes))
        else:
            print('x209: {}'.format(x209))
        x210=self.layernorm18(x209)
        if x210 is None:
            print('x210: {}'.format(x210))
        elif isinstance(x210, torch.Tensor):
            print('x210: {}'.format(x210.shape))
        elif isinstance(x210, tuple):
            tuple_shapes = '('
            for item in x210:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x210: {}'.format(tuple_shapes))
        else:
            print('x210: {}'.format(x210))
        x211=self.linear16(x210)
        if x211 is None:
            print('x211: {}'.format(x211))
        elif isinstance(x211, torch.Tensor):
            print('x211: {}'.format(x211.shape))
        elif isinstance(x211, tuple):
            tuple_shapes = '('
            for item in x211:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x211: {}'.format(tuple_shapes))
        else:
            print('x211: {}'.format(x211))
        x212=self.gelu7(x211)
        if x212 is None:
            print('x212: {}'.format(x212))
        elif isinstance(x212, torch.Tensor):
            print('x212: {}'.format(x212.shape))
        elif isinstance(x212, tuple):
            tuple_shapes = '('
            for item in x212:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x212: {}'.format(tuple_shapes))
        else:
            print('x212: {}'.format(x212))
        x213=self.dropout14(x212)
        if x213 is None:
            print('x213: {}'.format(x213))
        elif isinstance(x213, torch.Tensor):
            print('x213: {}'.format(x213.shape))
        elif isinstance(x213, tuple):
            tuple_shapes = '('
            for item in x213:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x213: {}'.format(tuple_shapes))
        else:
            print('x213: {}'.format(x213))
        x214=self.linear17(x213)
        if x214 is None:
            print('x214: {}'.format(x214))
        elif isinstance(x214, torch.Tensor):
            print('x214: {}'.format(x214.shape))
        elif isinstance(x214, tuple):
            tuple_shapes = '('
            for item in x214:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x214: {}'.format(tuple_shapes))
        else:
            print('x214: {}'.format(x214))
        x215=self.dropout15(x214)
        if x215 is None:
            print('x215: {}'.format(x215))
        elif isinstance(x215, torch.Tensor):
            print('x215: {}'.format(x215.shape))
        elif isinstance(x215, tuple):
            tuple_shapes = '('
            for item in x215:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x215: {}'.format(tuple_shapes))
        else:
            print('x215: {}'.format(x215))
        x216=stochastic_depth(x215, 0.15217391304347827, 'row', False)
        if x216 is None:
            print('x216: {}'.format(x216))
        elif isinstance(x216, torch.Tensor):
            print('x216: {}'.format(x216.shape))
        elif isinstance(x216, tuple):
            tuple_shapes = '('
            for item in x216:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x216: {}'.format(tuple_shapes))
        else:
            print('x216: {}'.format(x216))
        x217=operator.add(x209, x216)
        if x217 is None:
            print('x217: {}'.format(x217))
        elif isinstance(x217, torch.Tensor):
            print('x217: {}'.format(x217.shape))
        elif isinstance(x217, tuple):
            tuple_shapes = '('
            for item in x217:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x217: {}'.format(tuple_shapes))
        else:
            print('x217: {}'.format(x217))
        x218=self.layernorm19(x217)
        if x218 is None:
            print('x218: {}'.format(x218))
        elif isinstance(x218, torch.Tensor):
            print('x218: {}'.format(x218.shape))
        elif isinstance(x218, tuple):
            tuple_shapes = '('
            for item in x218:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x218: {}'.format(tuple_shapes))
        else:
            print('x218: {}'.format(x218))
        x221=operator.getitem(self.relative_position_bias_table8, self.relative_position_index8)
        if x221 is None:
            print('x221: {}'.format(x221))
        elif isinstance(x221, torch.Tensor):
            print('x221: {}'.format(x221.shape))
        elif isinstance(x221, tuple):
            tuple_shapes = '('
            for item in x221:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x221: {}'.format(tuple_shapes))
        else:
            print('x221: {}'.format(x221))
        x222=x221.view(49, 49, -1)
        if x222 is None:
            print('x222: {}'.format(x222))
        elif isinstance(x222, torch.Tensor):
            print('x222: {}'.format(x222.shape))
        elif isinstance(x222, tuple):
            tuple_shapes = '('
            for item in x222:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x222: {}'.format(tuple_shapes))
        else:
            print('x222: {}'.format(x222))
        x223=x222.permute(2, 0, 1)
        if x223 is None:
            print('x223: {}'.format(x223))
        elif isinstance(x223, torch.Tensor):
            print('x223: {}'.format(x223.shape))
        elif isinstance(x223, tuple):
            tuple_shapes = '('
            for item in x223:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x223: {}'.format(tuple_shapes))
        else:
            print('x223: {}'.format(x223))
        x224=x223.contiguous()
        if x224 is None:
            print('x224: {}'.format(x224))
        elif isinstance(x224, torch.Tensor):
            print('x224: {}'.format(x224.shape))
        elif isinstance(x224, tuple):
            tuple_shapes = '('
            for item in x224:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x224: {}'.format(tuple_shapes))
        else:
            print('x224: {}'.format(x224))
        x225=x224.unsqueeze(0)
        if x225 is None:
            print('x225: {}'.format(x225))
        elif isinstance(x225, torch.Tensor):
            print('x225: {}'.format(x225.shape))
        elif isinstance(x225, tuple):
            tuple_shapes = '('
            for item in x225:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x225: {}'.format(tuple_shapes))
        else:
            print('x225: {}'.format(x225))
        x230=torchvision.models.swin_transformer.shifted_window_attention(x218, self.weight16, self.weight17, x225, [7, 7], 16,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias16, proj_bias=self.bias17)
        if x230 is None:
            print('x230: {}'.format(x230))
        elif isinstance(x230, torch.Tensor):
            print('x230: {}'.format(x230.shape))
        elif isinstance(x230, tuple):
            tuple_shapes = '('
            for item in x230:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x230: {}'.format(tuple_shapes))
        else:
            print('x230: {}'.format(x230))
        x231=stochastic_depth(x230, 0.17391304347826086, 'row', False)
        if x231 is None:
            print('x231: {}'.format(x231))
        elif isinstance(x231, torch.Tensor):
            print('x231: {}'.format(x231.shape))
        elif isinstance(x231, tuple):
            tuple_shapes = '('
            for item in x231:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x231: {}'.format(tuple_shapes))
        else:
            print('x231: {}'.format(x231))
        x232=operator.add(x217, x231)
        if x232 is None:
            print('x232: {}'.format(x232))
        elif isinstance(x232, torch.Tensor):
            print('x232: {}'.format(x232.shape))
        elif isinstance(x232, tuple):
            tuple_shapes = '('
            for item in x232:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x232: {}'.format(tuple_shapes))
        else:
            print('x232: {}'.format(x232))
        x233=self.layernorm20(x232)
        if x233 is None:
            print('x233: {}'.format(x233))
        elif isinstance(x233, torch.Tensor):
            print('x233: {}'.format(x233.shape))
        elif isinstance(x233, tuple):
            tuple_shapes = '('
            for item in x233:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x233: {}'.format(tuple_shapes))
        else:
            print('x233: {}'.format(x233))
        x234=self.linear18(x233)
        if x234 is None:
            print('x234: {}'.format(x234))
        elif isinstance(x234, torch.Tensor):
            print('x234: {}'.format(x234.shape))
        elif isinstance(x234, tuple):
            tuple_shapes = '('
            for item in x234:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x234: {}'.format(tuple_shapes))
        else:
            print('x234: {}'.format(x234))
        x235=self.gelu8(x234)
        if x235 is None:
            print('x235: {}'.format(x235))
        elif isinstance(x235, torch.Tensor):
            print('x235: {}'.format(x235.shape))
        elif isinstance(x235, tuple):
            tuple_shapes = '('
            for item in x235:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x235: {}'.format(tuple_shapes))
        else:
            print('x235: {}'.format(x235))
        x236=self.dropout16(x235)
        if x236 is None:
            print('x236: {}'.format(x236))
        elif isinstance(x236, torch.Tensor):
            print('x236: {}'.format(x236.shape))
        elif isinstance(x236, tuple):
            tuple_shapes = '('
            for item in x236:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x236: {}'.format(tuple_shapes))
        else:
            print('x236: {}'.format(x236))
        x237=self.linear19(x236)
        if x237 is None:
            print('x237: {}'.format(x237))
        elif isinstance(x237, torch.Tensor):
            print('x237: {}'.format(x237.shape))
        elif isinstance(x237, tuple):
            tuple_shapes = '('
            for item in x237:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x237: {}'.format(tuple_shapes))
        else:
            print('x237: {}'.format(x237))
        x238=self.dropout17(x237)
        if x238 is None:
            print('x238: {}'.format(x238))
        elif isinstance(x238, torch.Tensor):
            print('x238: {}'.format(x238.shape))
        elif isinstance(x238, tuple):
            tuple_shapes = '('
            for item in x238:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x238: {}'.format(tuple_shapes))
        else:
            print('x238: {}'.format(x238))
        x239=stochastic_depth(x238, 0.17391304347826086, 'row', False)
        if x239 is None:
            print('x239: {}'.format(x239))
        elif isinstance(x239, torch.Tensor):
            print('x239: {}'.format(x239.shape))
        elif isinstance(x239, tuple):
            tuple_shapes = '('
            for item in x239:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x239: {}'.format(tuple_shapes))
        else:
            print('x239: {}'.format(x239))
        x240=operator.add(x232, x239)
        if x240 is None:
            print('x240: {}'.format(x240))
        elif isinstance(x240, torch.Tensor):
            print('x240: {}'.format(x240.shape))
        elif isinstance(x240, tuple):
            tuple_shapes = '('
            for item in x240:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x240: {}'.format(tuple_shapes))
        else:
            print('x240: {}'.format(x240))
        x241=self.layernorm21(x240)
        if x241 is None:
            print('x241: {}'.format(x241))
        elif isinstance(x241, torch.Tensor):
            print('x241: {}'.format(x241.shape))
        elif isinstance(x241, tuple):
            tuple_shapes = '('
            for item in x241:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x241: {}'.format(tuple_shapes))
        else:
            print('x241: {}'.format(x241))
        x244=operator.getitem(self.relative_position_bias_table9, self.relative_position_index9)
        if x244 is None:
            print('x244: {}'.format(x244))
        elif isinstance(x244, torch.Tensor):
            print('x244: {}'.format(x244.shape))
        elif isinstance(x244, tuple):
            tuple_shapes = '('
            for item in x244:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x244: {}'.format(tuple_shapes))
        else:
            print('x244: {}'.format(x244))
        x245=x244.view(49, 49, -1)
        if x245 is None:
            print('x245: {}'.format(x245))
        elif isinstance(x245, torch.Tensor):
            print('x245: {}'.format(x245.shape))
        elif isinstance(x245, tuple):
            tuple_shapes = '('
            for item in x245:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x245: {}'.format(tuple_shapes))
        else:
            print('x245: {}'.format(x245))
        x246=x245.permute(2, 0, 1)
        if x246 is None:
            print('x246: {}'.format(x246))
        elif isinstance(x246, torch.Tensor):
            print('x246: {}'.format(x246.shape))
        elif isinstance(x246, tuple):
            tuple_shapes = '('
            for item in x246:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x246: {}'.format(tuple_shapes))
        else:
            print('x246: {}'.format(x246))
        x247=x246.contiguous()
        if x247 is None:
            print('x247: {}'.format(x247))
        elif isinstance(x247, torch.Tensor):
            print('x247: {}'.format(x247.shape))
        elif isinstance(x247, tuple):
            tuple_shapes = '('
            for item in x247:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x247: {}'.format(tuple_shapes))
        else:
            print('x247: {}'.format(x247))
        x248=x247.unsqueeze(0)
        if x248 is None:
            print('x248: {}'.format(x248))
        elif isinstance(x248, torch.Tensor):
            print('x248: {}'.format(x248.shape))
        elif isinstance(x248, tuple):
            tuple_shapes = '('
            for item in x248:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x248: {}'.format(tuple_shapes))
        else:
            print('x248: {}'.format(x248))
        x253=torchvision.models.swin_transformer.shifted_window_attention(x241, self.weight18, self.weight19, x248, [7, 7], 16,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias18, proj_bias=self.bias19)
        if x253 is None:
            print('x253: {}'.format(x253))
        elif isinstance(x253, torch.Tensor):
            print('x253: {}'.format(x253.shape))
        elif isinstance(x253, tuple):
            tuple_shapes = '('
            for item in x253:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x253: {}'.format(tuple_shapes))
        else:
            print('x253: {}'.format(x253))
        x254=stochastic_depth(x253, 0.1956521739130435, 'row', False)
        if x254 is None:
            print('x254: {}'.format(x254))
        elif isinstance(x254, torch.Tensor):
            print('x254: {}'.format(x254.shape))
        elif isinstance(x254, tuple):
            tuple_shapes = '('
            for item in x254:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x254: {}'.format(tuple_shapes))
        else:
            print('x254: {}'.format(x254))
        x255=operator.add(x240, x254)
        if x255 is None:
            print('x255: {}'.format(x255))
        elif isinstance(x255, torch.Tensor):
            print('x255: {}'.format(x255.shape))
        elif isinstance(x255, tuple):
            tuple_shapes = '('
            for item in x255:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x255: {}'.format(tuple_shapes))
        else:
            print('x255: {}'.format(x255))
        x256=self.layernorm22(x255)
        if x256 is None:
            print('x256: {}'.format(x256))
        elif isinstance(x256, torch.Tensor):
            print('x256: {}'.format(x256.shape))
        elif isinstance(x256, tuple):
            tuple_shapes = '('
            for item in x256:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x256: {}'.format(tuple_shapes))
        else:
            print('x256: {}'.format(x256))
        x257=self.linear20(x256)
        if x257 is None:
            print('x257: {}'.format(x257))
        elif isinstance(x257, torch.Tensor):
            print('x257: {}'.format(x257.shape))
        elif isinstance(x257, tuple):
            tuple_shapes = '('
            for item in x257:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x257: {}'.format(tuple_shapes))
        else:
            print('x257: {}'.format(x257))
        x258=self.gelu9(x257)
        if x258 is None:
            print('x258: {}'.format(x258))
        elif isinstance(x258, torch.Tensor):
            print('x258: {}'.format(x258.shape))
        elif isinstance(x258, tuple):
            tuple_shapes = '('
            for item in x258:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x258: {}'.format(tuple_shapes))
        else:
            print('x258: {}'.format(x258))
        x259=self.dropout18(x258)
        if x259 is None:
            print('x259: {}'.format(x259))
        elif isinstance(x259, torch.Tensor):
            print('x259: {}'.format(x259.shape))
        elif isinstance(x259, tuple):
            tuple_shapes = '('
            for item in x259:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x259: {}'.format(tuple_shapes))
        else:
            print('x259: {}'.format(x259))
        x260=self.linear21(x259)
        if x260 is None:
            print('x260: {}'.format(x260))
        elif isinstance(x260, torch.Tensor):
            print('x260: {}'.format(x260.shape))
        elif isinstance(x260, tuple):
            tuple_shapes = '('
            for item in x260:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x260: {}'.format(tuple_shapes))
        else:
            print('x260: {}'.format(x260))
        x261=self.dropout19(x260)
        if x261 is None:
            print('x261: {}'.format(x261))
        elif isinstance(x261, torch.Tensor):
            print('x261: {}'.format(x261.shape))
        elif isinstance(x261, tuple):
            tuple_shapes = '('
            for item in x261:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x261: {}'.format(tuple_shapes))
        else:
            print('x261: {}'.format(x261))
        x262=stochastic_depth(x261, 0.1956521739130435, 'row', False)
        if x262 is None:
            print('x262: {}'.format(x262))
        elif isinstance(x262, torch.Tensor):
            print('x262: {}'.format(x262.shape))
        elif isinstance(x262, tuple):
            tuple_shapes = '('
            for item in x262:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x262: {}'.format(tuple_shapes))
        else:
            print('x262: {}'.format(x262))
        x263=operator.add(x255, x262)
        if x263 is None:
            print('x263: {}'.format(x263))
        elif isinstance(x263, torch.Tensor):
            print('x263: {}'.format(x263.shape))
        elif isinstance(x263, tuple):
            tuple_shapes = '('
            for item in x263:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x263: {}'.format(tuple_shapes))
        else:
            print('x263: {}'.format(x263))
        x264=self.layernorm23(x263)
        if x264 is None:
            print('x264: {}'.format(x264))
        elif isinstance(x264, torch.Tensor):
            print('x264: {}'.format(x264.shape))
        elif isinstance(x264, tuple):
            tuple_shapes = '('
            for item in x264:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x264: {}'.format(tuple_shapes))
        else:
            print('x264: {}'.format(x264))
        x267=operator.getitem(self.relative_position_bias_table10, self.relative_position_index10)
        if x267 is None:
            print('x267: {}'.format(x267))
        elif isinstance(x267, torch.Tensor):
            print('x267: {}'.format(x267.shape))
        elif isinstance(x267, tuple):
            tuple_shapes = '('
            for item in x267:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x267: {}'.format(tuple_shapes))
        else:
            print('x267: {}'.format(x267))
        x268=x267.view(49, 49, -1)
        if x268 is None:
            print('x268: {}'.format(x268))
        elif isinstance(x268, torch.Tensor):
            print('x268: {}'.format(x268.shape))
        elif isinstance(x268, tuple):
            tuple_shapes = '('
            for item in x268:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x268: {}'.format(tuple_shapes))
        else:
            print('x268: {}'.format(x268))
        x269=x268.permute(2, 0, 1)
        if x269 is None:
            print('x269: {}'.format(x269))
        elif isinstance(x269, torch.Tensor):
            print('x269: {}'.format(x269.shape))
        elif isinstance(x269, tuple):
            tuple_shapes = '('
            for item in x269:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x269: {}'.format(tuple_shapes))
        else:
            print('x269: {}'.format(x269))
        x270=x269.contiguous()
        if x270 is None:
            print('x270: {}'.format(x270))
        elif isinstance(x270, torch.Tensor):
            print('x270: {}'.format(x270.shape))
        elif isinstance(x270, tuple):
            tuple_shapes = '('
            for item in x270:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x270: {}'.format(tuple_shapes))
        else:
            print('x270: {}'.format(x270))
        x271=x270.unsqueeze(0)
        if x271 is None:
            print('x271: {}'.format(x271))
        elif isinstance(x271, torch.Tensor):
            print('x271: {}'.format(x271.shape))
        elif isinstance(x271, tuple):
            tuple_shapes = '('
            for item in x271:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x271: {}'.format(tuple_shapes))
        else:
            print('x271: {}'.format(x271))
        x276=torchvision.models.swin_transformer.shifted_window_attention(x264, self.weight20, self.weight21, x271, [7, 7], 16,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias20, proj_bias=self.bias21)
        if x276 is None:
            print('x276: {}'.format(x276))
        elif isinstance(x276, torch.Tensor):
            print('x276: {}'.format(x276.shape))
        elif isinstance(x276, tuple):
            tuple_shapes = '('
            for item in x276:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x276: {}'.format(tuple_shapes))
        else:
            print('x276: {}'.format(x276))
        x277=stochastic_depth(x276, 0.21739130434782608, 'row', False)
        if x277 is None:
            print('x277: {}'.format(x277))
        elif isinstance(x277, torch.Tensor):
            print('x277: {}'.format(x277.shape))
        elif isinstance(x277, tuple):
            tuple_shapes = '('
            for item in x277:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x277: {}'.format(tuple_shapes))
        else:
            print('x277: {}'.format(x277))
        x278=operator.add(x263, x277)
        if x278 is None:
            print('x278: {}'.format(x278))
        elif isinstance(x278, torch.Tensor):
            print('x278: {}'.format(x278.shape))
        elif isinstance(x278, tuple):
            tuple_shapes = '('
            for item in x278:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x278: {}'.format(tuple_shapes))
        else:
            print('x278: {}'.format(x278))
        x279=self.layernorm24(x278)
        if x279 is None:
            print('x279: {}'.format(x279))
        elif isinstance(x279, torch.Tensor):
            print('x279: {}'.format(x279.shape))
        elif isinstance(x279, tuple):
            tuple_shapes = '('
            for item in x279:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x279: {}'.format(tuple_shapes))
        else:
            print('x279: {}'.format(x279))
        x280=self.linear22(x279)
        if x280 is None:
            print('x280: {}'.format(x280))
        elif isinstance(x280, torch.Tensor):
            print('x280: {}'.format(x280.shape))
        elif isinstance(x280, tuple):
            tuple_shapes = '('
            for item in x280:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x280: {}'.format(tuple_shapes))
        else:
            print('x280: {}'.format(x280))
        x281=self.gelu10(x280)
        if x281 is None:
            print('x281: {}'.format(x281))
        elif isinstance(x281, torch.Tensor):
            print('x281: {}'.format(x281.shape))
        elif isinstance(x281, tuple):
            tuple_shapes = '('
            for item in x281:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x281: {}'.format(tuple_shapes))
        else:
            print('x281: {}'.format(x281))
        x282=self.dropout20(x281)
        if x282 is None:
            print('x282: {}'.format(x282))
        elif isinstance(x282, torch.Tensor):
            print('x282: {}'.format(x282.shape))
        elif isinstance(x282, tuple):
            tuple_shapes = '('
            for item in x282:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x282: {}'.format(tuple_shapes))
        else:
            print('x282: {}'.format(x282))
        x283=self.linear23(x282)
        if x283 is None:
            print('x283: {}'.format(x283))
        elif isinstance(x283, torch.Tensor):
            print('x283: {}'.format(x283.shape))
        elif isinstance(x283, tuple):
            tuple_shapes = '('
            for item in x283:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x283: {}'.format(tuple_shapes))
        else:
            print('x283: {}'.format(x283))
        x284=self.dropout21(x283)
        if x284 is None:
            print('x284: {}'.format(x284))
        elif isinstance(x284, torch.Tensor):
            print('x284: {}'.format(x284.shape))
        elif isinstance(x284, tuple):
            tuple_shapes = '('
            for item in x284:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x284: {}'.format(tuple_shapes))
        else:
            print('x284: {}'.format(x284))
        x285=stochastic_depth(x284, 0.21739130434782608, 'row', False)
        if x285 is None:
            print('x285: {}'.format(x285))
        elif isinstance(x285, torch.Tensor):
            print('x285: {}'.format(x285.shape))
        elif isinstance(x285, tuple):
            tuple_shapes = '('
            for item in x285:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x285: {}'.format(tuple_shapes))
        else:
            print('x285: {}'.format(x285))
        x286=operator.add(x278, x285)
        if x286 is None:
            print('x286: {}'.format(x286))
        elif isinstance(x286, torch.Tensor):
            print('x286: {}'.format(x286.shape))
        elif isinstance(x286, tuple):
            tuple_shapes = '('
            for item in x286:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x286: {}'.format(tuple_shapes))
        else:
            print('x286: {}'.format(x286))
        x287=self.layernorm25(x286)
        if x287 is None:
            print('x287: {}'.format(x287))
        elif isinstance(x287, torch.Tensor):
            print('x287: {}'.format(x287.shape))
        elif isinstance(x287, tuple):
            tuple_shapes = '('
            for item in x287:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x287: {}'.format(tuple_shapes))
        else:
            print('x287: {}'.format(x287))
        x290=operator.getitem(self.relative_position_bias_table11, self.relative_position_index11)
        if x290 is None:
            print('x290: {}'.format(x290))
        elif isinstance(x290, torch.Tensor):
            print('x290: {}'.format(x290.shape))
        elif isinstance(x290, tuple):
            tuple_shapes = '('
            for item in x290:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x290: {}'.format(tuple_shapes))
        else:
            print('x290: {}'.format(x290))
        x291=x290.view(49, 49, -1)
        if x291 is None:
            print('x291: {}'.format(x291))
        elif isinstance(x291, torch.Tensor):
            print('x291: {}'.format(x291.shape))
        elif isinstance(x291, tuple):
            tuple_shapes = '('
            for item in x291:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x291: {}'.format(tuple_shapes))
        else:
            print('x291: {}'.format(x291))
        x292=x291.permute(2, 0, 1)
        if x292 is None:
            print('x292: {}'.format(x292))
        elif isinstance(x292, torch.Tensor):
            print('x292: {}'.format(x292.shape))
        elif isinstance(x292, tuple):
            tuple_shapes = '('
            for item in x292:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x292: {}'.format(tuple_shapes))
        else:
            print('x292: {}'.format(x292))
        x293=x292.contiguous()
        if x293 is None:
            print('x293: {}'.format(x293))
        elif isinstance(x293, torch.Tensor):
            print('x293: {}'.format(x293.shape))
        elif isinstance(x293, tuple):
            tuple_shapes = '('
            for item in x293:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x293: {}'.format(tuple_shapes))
        else:
            print('x293: {}'.format(x293))
        x294=x293.unsqueeze(0)
        if x294 is None:
            print('x294: {}'.format(x294))
        elif isinstance(x294, torch.Tensor):
            print('x294: {}'.format(x294.shape))
        elif isinstance(x294, tuple):
            tuple_shapes = '('
            for item in x294:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x294: {}'.format(tuple_shapes))
        else:
            print('x294: {}'.format(x294))
        x299=torchvision.models.swin_transformer.shifted_window_attention(x287, self.weight22, self.weight23, x294, [7, 7], 16,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias22, proj_bias=self.bias23)
        if x299 is None:
            print('x299: {}'.format(x299))
        elif isinstance(x299, torch.Tensor):
            print('x299: {}'.format(x299.shape))
        elif isinstance(x299, tuple):
            tuple_shapes = '('
            for item in x299:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x299: {}'.format(tuple_shapes))
        else:
            print('x299: {}'.format(x299))
        x300=stochastic_depth(x299, 0.2391304347826087, 'row', False)
        if x300 is None:
            print('x300: {}'.format(x300))
        elif isinstance(x300, torch.Tensor):
            print('x300: {}'.format(x300.shape))
        elif isinstance(x300, tuple):
            tuple_shapes = '('
            for item in x300:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x300: {}'.format(tuple_shapes))
        else:
            print('x300: {}'.format(x300))
        x301=operator.add(x286, x300)
        if x301 is None:
            print('x301: {}'.format(x301))
        elif isinstance(x301, torch.Tensor):
            print('x301: {}'.format(x301.shape))
        elif isinstance(x301, tuple):
            tuple_shapes = '('
            for item in x301:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x301: {}'.format(tuple_shapes))
        else:
            print('x301: {}'.format(x301))
        x302=self.layernorm26(x301)
        if x302 is None:
            print('x302: {}'.format(x302))
        elif isinstance(x302, torch.Tensor):
            print('x302: {}'.format(x302.shape))
        elif isinstance(x302, tuple):
            tuple_shapes = '('
            for item in x302:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x302: {}'.format(tuple_shapes))
        else:
            print('x302: {}'.format(x302))
        x303=self.linear24(x302)
        if x303 is None:
            print('x303: {}'.format(x303))
        elif isinstance(x303, torch.Tensor):
            print('x303: {}'.format(x303.shape))
        elif isinstance(x303, tuple):
            tuple_shapes = '('
            for item in x303:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x303: {}'.format(tuple_shapes))
        else:
            print('x303: {}'.format(x303))
        x304=self.gelu11(x303)
        if x304 is None:
            print('x304: {}'.format(x304))
        elif isinstance(x304, torch.Tensor):
            print('x304: {}'.format(x304.shape))
        elif isinstance(x304, tuple):
            tuple_shapes = '('
            for item in x304:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x304: {}'.format(tuple_shapes))
        else:
            print('x304: {}'.format(x304))
        x305=self.dropout22(x304)
        if x305 is None:
            print('x305: {}'.format(x305))
        elif isinstance(x305, torch.Tensor):
            print('x305: {}'.format(x305.shape))
        elif isinstance(x305, tuple):
            tuple_shapes = '('
            for item in x305:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x305: {}'.format(tuple_shapes))
        else:
            print('x305: {}'.format(x305))
        x306=self.linear25(x305)
        if x306 is None:
            print('x306: {}'.format(x306))
        elif isinstance(x306, torch.Tensor):
            print('x306: {}'.format(x306.shape))
        elif isinstance(x306, tuple):
            tuple_shapes = '('
            for item in x306:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x306: {}'.format(tuple_shapes))
        else:
            print('x306: {}'.format(x306))
        x307=self.dropout23(x306)
        if x307 is None:
            print('x307: {}'.format(x307))
        elif isinstance(x307, torch.Tensor):
            print('x307: {}'.format(x307.shape))
        elif isinstance(x307, tuple):
            tuple_shapes = '('
            for item in x307:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x307: {}'.format(tuple_shapes))
        else:
            print('x307: {}'.format(x307))
        x308=stochastic_depth(x307, 0.2391304347826087, 'row', False)
        if x308 is None:
            print('x308: {}'.format(x308))
        elif isinstance(x308, torch.Tensor):
            print('x308: {}'.format(x308.shape))
        elif isinstance(x308, tuple):
            tuple_shapes = '('
            for item in x308:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x308: {}'.format(tuple_shapes))
        else:
            print('x308: {}'.format(x308))
        x309=operator.add(x301, x308)
        if x309 is None:
            print('x309: {}'.format(x309))
        elif isinstance(x309, torch.Tensor):
            print('x309: {}'.format(x309.shape))
        elif isinstance(x309, tuple):
            tuple_shapes = '('
            for item in x309:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x309: {}'.format(tuple_shapes))
        else:
            print('x309: {}'.format(x309))
        x310=self.layernorm27(x309)
        if x310 is None:
            print('x310: {}'.format(x310))
        elif isinstance(x310, torch.Tensor):
            print('x310: {}'.format(x310.shape))
        elif isinstance(x310, tuple):
            tuple_shapes = '('
            for item in x310:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x310: {}'.format(tuple_shapes))
        else:
            print('x310: {}'.format(x310))
        x313=operator.getitem(self.relative_position_bias_table12, self.relative_position_index12)
        if x313 is None:
            print('x313: {}'.format(x313))
        elif isinstance(x313, torch.Tensor):
            print('x313: {}'.format(x313.shape))
        elif isinstance(x313, tuple):
            tuple_shapes = '('
            for item in x313:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x313: {}'.format(tuple_shapes))
        else:
            print('x313: {}'.format(x313))
        x314=x313.view(49, 49, -1)
        if x314 is None:
            print('x314: {}'.format(x314))
        elif isinstance(x314, torch.Tensor):
            print('x314: {}'.format(x314.shape))
        elif isinstance(x314, tuple):
            tuple_shapes = '('
            for item in x314:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x314: {}'.format(tuple_shapes))
        else:
            print('x314: {}'.format(x314))
        x315=x314.permute(2, 0, 1)
        if x315 is None:
            print('x315: {}'.format(x315))
        elif isinstance(x315, torch.Tensor):
            print('x315: {}'.format(x315.shape))
        elif isinstance(x315, tuple):
            tuple_shapes = '('
            for item in x315:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x315: {}'.format(tuple_shapes))
        else:
            print('x315: {}'.format(x315))
        x316=x315.contiguous()
        if x316 is None:
            print('x316: {}'.format(x316))
        elif isinstance(x316, torch.Tensor):
            print('x316: {}'.format(x316.shape))
        elif isinstance(x316, tuple):
            tuple_shapes = '('
            for item in x316:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x316: {}'.format(tuple_shapes))
        else:
            print('x316: {}'.format(x316))
        x317=x316.unsqueeze(0)
        if x317 is None:
            print('x317: {}'.format(x317))
        elif isinstance(x317, torch.Tensor):
            print('x317: {}'.format(x317.shape))
        elif isinstance(x317, tuple):
            tuple_shapes = '('
            for item in x317:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x317: {}'.format(tuple_shapes))
        else:
            print('x317: {}'.format(x317))
        x322=torchvision.models.swin_transformer.shifted_window_attention(x310, self.weight24, self.weight25, x317, [7, 7], 16,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias24, proj_bias=self.bias25)
        if x322 is None:
            print('x322: {}'.format(x322))
        elif isinstance(x322, torch.Tensor):
            print('x322: {}'.format(x322.shape))
        elif isinstance(x322, tuple):
            tuple_shapes = '('
            for item in x322:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x322: {}'.format(tuple_shapes))
        else:
            print('x322: {}'.format(x322))
        x323=stochastic_depth(x322, 0.2608695652173913, 'row', False)
        if x323 is None:
            print('x323: {}'.format(x323))
        elif isinstance(x323, torch.Tensor):
            print('x323: {}'.format(x323.shape))
        elif isinstance(x323, tuple):
            tuple_shapes = '('
            for item in x323:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x323: {}'.format(tuple_shapes))
        else:
            print('x323: {}'.format(x323))
        x324=operator.add(x309, x323)
        if x324 is None:
            print('x324: {}'.format(x324))
        elif isinstance(x324, torch.Tensor):
            print('x324: {}'.format(x324.shape))
        elif isinstance(x324, tuple):
            tuple_shapes = '('
            for item in x324:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x324: {}'.format(tuple_shapes))
        else:
            print('x324: {}'.format(x324))
        x325=self.layernorm28(x324)
        if x325 is None:
            print('x325: {}'.format(x325))
        elif isinstance(x325, torch.Tensor):
            print('x325: {}'.format(x325.shape))
        elif isinstance(x325, tuple):
            tuple_shapes = '('
            for item in x325:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x325: {}'.format(tuple_shapes))
        else:
            print('x325: {}'.format(x325))
        x326=self.linear26(x325)
        if x326 is None:
            print('x326: {}'.format(x326))
        elif isinstance(x326, torch.Tensor):
            print('x326: {}'.format(x326.shape))
        elif isinstance(x326, tuple):
            tuple_shapes = '('
            for item in x326:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x326: {}'.format(tuple_shapes))
        else:
            print('x326: {}'.format(x326))
        x327=self.gelu12(x326)
        if x327 is None:
            print('x327: {}'.format(x327))
        elif isinstance(x327, torch.Tensor):
            print('x327: {}'.format(x327.shape))
        elif isinstance(x327, tuple):
            tuple_shapes = '('
            for item in x327:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x327: {}'.format(tuple_shapes))
        else:
            print('x327: {}'.format(x327))
        x328=self.dropout24(x327)
        if x328 is None:
            print('x328: {}'.format(x328))
        elif isinstance(x328, torch.Tensor):
            print('x328: {}'.format(x328.shape))
        elif isinstance(x328, tuple):
            tuple_shapes = '('
            for item in x328:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x328: {}'.format(tuple_shapes))
        else:
            print('x328: {}'.format(x328))
        x329=self.linear27(x328)
        if x329 is None:
            print('x329: {}'.format(x329))
        elif isinstance(x329, torch.Tensor):
            print('x329: {}'.format(x329.shape))
        elif isinstance(x329, tuple):
            tuple_shapes = '('
            for item in x329:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x329: {}'.format(tuple_shapes))
        else:
            print('x329: {}'.format(x329))
        x330=self.dropout25(x329)
        if x330 is None:
            print('x330: {}'.format(x330))
        elif isinstance(x330, torch.Tensor):
            print('x330: {}'.format(x330.shape))
        elif isinstance(x330, tuple):
            tuple_shapes = '('
            for item in x330:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x330: {}'.format(tuple_shapes))
        else:
            print('x330: {}'.format(x330))
        x331=stochastic_depth(x330, 0.2608695652173913, 'row', False)
        if x331 is None:
            print('x331: {}'.format(x331))
        elif isinstance(x331, torch.Tensor):
            print('x331: {}'.format(x331.shape))
        elif isinstance(x331, tuple):
            tuple_shapes = '('
            for item in x331:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x331: {}'.format(tuple_shapes))
        else:
            print('x331: {}'.format(x331))
        x332=operator.add(x324, x331)
        if x332 is None:
            print('x332: {}'.format(x332))
        elif isinstance(x332, torch.Tensor):
            print('x332: {}'.format(x332.shape))
        elif isinstance(x332, tuple):
            tuple_shapes = '('
            for item in x332:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x332: {}'.format(tuple_shapes))
        else:
            print('x332: {}'.format(x332))
        x333=self.layernorm29(x332)
        if x333 is None:
            print('x333: {}'.format(x333))
        elif isinstance(x333, torch.Tensor):
            print('x333: {}'.format(x333.shape))
        elif isinstance(x333, tuple):
            tuple_shapes = '('
            for item in x333:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x333: {}'.format(tuple_shapes))
        else:
            print('x333: {}'.format(x333))
        x336=operator.getitem(self.relative_position_bias_table13, self.relative_position_index13)
        if x336 is None:
            print('x336: {}'.format(x336))
        elif isinstance(x336, torch.Tensor):
            print('x336: {}'.format(x336.shape))
        elif isinstance(x336, tuple):
            tuple_shapes = '('
            for item in x336:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x336: {}'.format(tuple_shapes))
        else:
            print('x336: {}'.format(x336))
        x337=x336.view(49, 49, -1)
        if x337 is None:
            print('x337: {}'.format(x337))
        elif isinstance(x337, torch.Tensor):
            print('x337: {}'.format(x337.shape))
        elif isinstance(x337, tuple):
            tuple_shapes = '('
            for item in x337:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x337: {}'.format(tuple_shapes))
        else:
            print('x337: {}'.format(x337))
        x338=x337.permute(2, 0, 1)
        if x338 is None:
            print('x338: {}'.format(x338))
        elif isinstance(x338, torch.Tensor):
            print('x338: {}'.format(x338.shape))
        elif isinstance(x338, tuple):
            tuple_shapes = '('
            for item in x338:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x338: {}'.format(tuple_shapes))
        else:
            print('x338: {}'.format(x338))
        x339=x338.contiguous()
        if x339 is None:
            print('x339: {}'.format(x339))
        elif isinstance(x339, torch.Tensor):
            print('x339: {}'.format(x339.shape))
        elif isinstance(x339, tuple):
            tuple_shapes = '('
            for item in x339:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x339: {}'.format(tuple_shapes))
        else:
            print('x339: {}'.format(x339))
        x340=x339.unsqueeze(0)
        if x340 is None:
            print('x340: {}'.format(x340))
        elif isinstance(x340, torch.Tensor):
            print('x340: {}'.format(x340.shape))
        elif isinstance(x340, tuple):
            tuple_shapes = '('
            for item in x340:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x340: {}'.format(tuple_shapes))
        else:
            print('x340: {}'.format(x340))
        x345=torchvision.models.swin_transformer.shifted_window_attention(x333, self.weight26, self.weight27, x340, [7, 7], 16,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias26, proj_bias=self.bias27)
        if x345 is None:
            print('x345: {}'.format(x345))
        elif isinstance(x345, torch.Tensor):
            print('x345: {}'.format(x345.shape))
        elif isinstance(x345, tuple):
            tuple_shapes = '('
            for item in x345:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x345: {}'.format(tuple_shapes))
        else:
            print('x345: {}'.format(x345))
        x346=stochastic_depth(x345, 0.2826086956521739, 'row', False)
        if x346 is None:
            print('x346: {}'.format(x346))
        elif isinstance(x346, torch.Tensor):
            print('x346: {}'.format(x346.shape))
        elif isinstance(x346, tuple):
            tuple_shapes = '('
            for item in x346:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x346: {}'.format(tuple_shapes))
        else:
            print('x346: {}'.format(x346))
        x347=operator.add(x332, x346)
        if x347 is None:
            print('x347: {}'.format(x347))
        elif isinstance(x347, torch.Tensor):
            print('x347: {}'.format(x347.shape))
        elif isinstance(x347, tuple):
            tuple_shapes = '('
            for item in x347:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x347: {}'.format(tuple_shapes))
        else:
            print('x347: {}'.format(x347))
        x348=self.layernorm30(x347)
        if x348 is None:
            print('x348: {}'.format(x348))
        elif isinstance(x348, torch.Tensor):
            print('x348: {}'.format(x348.shape))
        elif isinstance(x348, tuple):
            tuple_shapes = '('
            for item in x348:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x348: {}'.format(tuple_shapes))
        else:
            print('x348: {}'.format(x348))
        x349=self.linear28(x348)
        if x349 is None:
            print('x349: {}'.format(x349))
        elif isinstance(x349, torch.Tensor):
            print('x349: {}'.format(x349.shape))
        elif isinstance(x349, tuple):
            tuple_shapes = '('
            for item in x349:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x349: {}'.format(tuple_shapes))
        else:
            print('x349: {}'.format(x349))
        x350=self.gelu13(x349)
        if x350 is None:
            print('x350: {}'.format(x350))
        elif isinstance(x350, torch.Tensor):
            print('x350: {}'.format(x350.shape))
        elif isinstance(x350, tuple):
            tuple_shapes = '('
            for item in x350:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x350: {}'.format(tuple_shapes))
        else:
            print('x350: {}'.format(x350))
        x351=self.dropout26(x350)
        if x351 is None:
            print('x351: {}'.format(x351))
        elif isinstance(x351, torch.Tensor):
            print('x351: {}'.format(x351.shape))
        elif isinstance(x351, tuple):
            tuple_shapes = '('
            for item in x351:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x351: {}'.format(tuple_shapes))
        else:
            print('x351: {}'.format(x351))
        x352=self.linear29(x351)
        if x352 is None:
            print('x352: {}'.format(x352))
        elif isinstance(x352, torch.Tensor):
            print('x352: {}'.format(x352.shape))
        elif isinstance(x352, tuple):
            tuple_shapes = '('
            for item in x352:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x352: {}'.format(tuple_shapes))
        else:
            print('x352: {}'.format(x352))
        x353=self.dropout27(x352)
        if x353 is None:
            print('x353: {}'.format(x353))
        elif isinstance(x353, torch.Tensor):
            print('x353: {}'.format(x353.shape))
        elif isinstance(x353, tuple):
            tuple_shapes = '('
            for item in x353:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x353: {}'.format(tuple_shapes))
        else:
            print('x353: {}'.format(x353))
        x354=stochastic_depth(x353, 0.2826086956521739, 'row', False)
        if x354 is None:
            print('x354: {}'.format(x354))
        elif isinstance(x354, torch.Tensor):
            print('x354: {}'.format(x354.shape))
        elif isinstance(x354, tuple):
            tuple_shapes = '('
            for item in x354:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x354: {}'.format(tuple_shapes))
        else:
            print('x354: {}'.format(x354))
        x355=operator.add(x347, x354)
        if x355 is None:
            print('x355: {}'.format(x355))
        elif isinstance(x355, torch.Tensor):
            print('x355: {}'.format(x355.shape))
        elif isinstance(x355, tuple):
            tuple_shapes = '('
            for item in x355:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x355: {}'.format(tuple_shapes))
        else:
            print('x355: {}'.format(x355))
        x356=self.layernorm31(x355)
        if x356 is None:
            print('x356: {}'.format(x356))
        elif isinstance(x356, torch.Tensor):
            print('x356: {}'.format(x356.shape))
        elif isinstance(x356, tuple):
            tuple_shapes = '('
            for item in x356:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x356: {}'.format(tuple_shapes))
        else:
            print('x356: {}'.format(x356))
        x359=operator.getitem(self.relative_position_bias_table14, self.relative_position_index14)
        if x359 is None:
            print('x359: {}'.format(x359))
        elif isinstance(x359, torch.Tensor):
            print('x359: {}'.format(x359.shape))
        elif isinstance(x359, tuple):
            tuple_shapes = '('
            for item in x359:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x359: {}'.format(tuple_shapes))
        else:
            print('x359: {}'.format(x359))
        x360=x359.view(49, 49, -1)
        if x360 is None:
            print('x360: {}'.format(x360))
        elif isinstance(x360, torch.Tensor):
            print('x360: {}'.format(x360.shape))
        elif isinstance(x360, tuple):
            tuple_shapes = '('
            for item in x360:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x360: {}'.format(tuple_shapes))
        else:
            print('x360: {}'.format(x360))
        x361=x360.permute(2, 0, 1)
        if x361 is None:
            print('x361: {}'.format(x361))
        elif isinstance(x361, torch.Tensor):
            print('x361: {}'.format(x361.shape))
        elif isinstance(x361, tuple):
            tuple_shapes = '('
            for item in x361:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x361: {}'.format(tuple_shapes))
        else:
            print('x361: {}'.format(x361))
        x362=x361.contiguous()
        if x362 is None:
            print('x362: {}'.format(x362))
        elif isinstance(x362, torch.Tensor):
            print('x362: {}'.format(x362.shape))
        elif isinstance(x362, tuple):
            tuple_shapes = '('
            for item in x362:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x362: {}'.format(tuple_shapes))
        else:
            print('x362: {}'.format(x362))
        x363=x362.unsqueeze(0)
        if x363 is None:
            print('x363: {}'.format(x363))
        elif isinstance(x363, torch.Tensor):
            print('x363: {}'.format(x363.shape))
        elif isinstance(x363, tuple):
            tuple_shapes = '('
            for item in x363:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x363: {}'.format(tuple_shapes))
        else:
            print('x363: {}'.format(x363))
        x368=torchvision.models.swin_transformer.shifted_window_attention(x356, self.weight28, self.weight29, x363, [7, 7], 16,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias28, proj_bias=self.bias29)
        if x368 is None:
            print('x368: {}'.format(x368))
        elif isinstance(x368, torch.Tensor):
            print('x368: {}'.format(x368.shape))
        elif isinstance(x368, tuple):
            tuple_shapes = '('
            for item in x368:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x368: {}'.format(tuple_shapes))
        else:
            print('x368: {}'.format(x368))
        x369=stochastic_depth(x368, 0.30434782608695654, 'row', False)
        if x369 is None:
            print('x369: {}'.format(x369))
        elif isinstance(x369, torch.Tensor):
            print('x369: {}'.format(x369.shape))
        elif isinstance(x369, tuple):
            tuple_shapes = '('
            for item in x369:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x369: {}'.format(tuple_shapes))
        else:
            print('x369: {}'.format(x369))
        x370=operator.add(x355, x369)
        if x370 is None:
            print('x370: {}'.format(x370))
        elif isinstance(x370, torch.Tensor):
            print('x370: {}'.format(x370.shape))
        elif isinstance(x370, tuple):
            tuple_shapes = '('
            for item in x370:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x370: {}'.format(tuple_shapes))
        else:
            print('x370: {}'.format(x370))
        x371=self.layernorm32(x370)
        if x371 is None:
            print('x371: {}'.format(x371))
        elif isinstance(x371, torch.Tensor):
            print('x371: {}'.format(x371.shape))
        elif isinstance(x371, tuple):
            tuple_shapes = '('
            for item in x371:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x371: {}'.format(tuple_shapes))
        else:
            print('x371: {}'.format(x371))
        x372=self.linear30(x371)
        if x372 is None:
            print('x372: {}'.format(x372))
        elif isinstance(x372, torch.Tensor):
            print('x372: {}'.format(x372.shape))
        elif isinstance(x372, tuple):
            tuple_shapes = '('
            for item in x372:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x372: {}'.format(tuple_shapes))
        else:
            print('x372: {}'.format(x372))
        x373=self.gelu14(x372)
        if x373 is None:
            print('x373: {}'.format(x373))
        elif isinstance(x373, torch.Tensor):
            print('x373: {}'.format(x373.shape))
        elif isinstance(x373, tuple):
            tuple_shapes = '('
            for item in x373:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x373: {}'.format(tuple_shapes))
        else:
            print('x373: {}'.format(x373))
        x374=self.dropout28(x373)
        if x374 is None:
            print('x374: {}'.format(x374))
        elif isinstance(x374, torch.Tensor):
            print('x374: {}'.format(x374.shape))
        elif isinstance(x374, tuple):
            tuple_shapes = '('
            for item in x374:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x374: {}'.format(tuple_shapes))
        else:
            print('x374: {}'.format(x374))
        x375=self.linear31(x374)
        if x375 is None:
            print('x375: {}'.format(x375))
        elif isinstance(x375, torch.Tensor):
            print('x375: {}'.format(x375.shape))
        elif isinstance(x375, tuple):
            tuple_shapes = '('
            for item in x375:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x375: {}'.format(tuple_shapes))
        else:
            print('x375: {}'.format(x375))
        x376=self.dropout29(x375)
        if x376 is None:
            print('x376: {}'.format(x376))
        elif isinstance(x376, torch.Tensor):
            print('x376: {}'.format(x376.shape))
        elif isinstance(x376, tuple):
            tuple_shapes = '('
            for item in x376:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x376: {}'.format(tuple_shapes))
        else:
            print('x376: {}'.format(x376))
        x377=stochastic_depth(x376, 0.30434782608695654, 'row', False)
        if x377 is None:
            print('x377: {}'.format(x377))
        elif isinstance(x377, torch.Tensor):
            print('x377: {}'.format(x377.shape))
        elif isinstance(x377, tuple):
            tuple_shapes = '('
            for item in x377:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x377: {}'.format(tuple_shapes))
        else:
            print('x377: {}'.format(x377))
        x378=operator.add(x370, x377)
        if x378 is None:
            print('x378: {}'.format(x378))
        elif isinstance(x378, torch.Tensor):
            print('x378: {}'.format(x378.shape))
        elif isinstance(x378, tuple):
            tuple_shapes = '('
            for item in x378:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x378: {}'.format(tuple_shapes))
        else:
            print('x378: {}'.format(x378))
        x379=self.layernorm33(x378)
        if x379 is None:
            print('x379: {}'.format(x379))
        elif isinstance(x379, torch.Tensor):
            print('x379: {}'.format(x379.shape))
        elif isinstance(x379, tuple):
            tuple_shapes = '('
            for item in x379:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x379: {}'.format(tuple_shapes))
        else:
            print('x379: {}'.format(x379))
        x382=operator.getitem(self.relative_position_bias_table15, self.relative_position_index15)
        if x382 is None:
            print('x382: {}'.format(x382))
        elif isinstance(x382, torch.Tensor):
            print('x382: {}'.format(x382.shape))
        elif isinstance(x382, tuple):
            tuple_shapes = '('
            for item in x382:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x382: {}'.format(tuple_shapes))
        else:
            print('x382: {}'.format(x382))
        x383=x382.view(49, 49, -1)
        if x383 is None:
            print('x383: {}'.format(x383))
        elif isinstance(x383, torch.Tensor):
            print('x383: {}'.format(x383.shape))
        elif isinstance(x383, tuple):
            tuple_shapes = '('
            for item in x383:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x383: {}'.format(tuple_shapes))
        else:
            print('x383: {}'.format(x383))
        x384=x383.permute(2, 0, 1)
        if x384 is None:
            print('x384: {}'.format(x384))
        elif isinstance(x384, torch.Tensor):
            print('x384: {}'.format(x384.shape))
        elif isinstance(x384, tuple):
            tuple_shapes = '('
            for item in x384:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x384: {}'.format(tuple_shapes))
        else:
            print('x384: {}'.format(x384))
        x385=x384.contiguous()
        if x385 is None:
            print('x385: {}'.format(x385))
        elif isinstance(x385, torch.Tensor):
            print('x385: {}'.format(x385.shape))
        elif isinstance(x385, tuple):
            tuple_shapes = '('
            for item in x385:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x385: {}'.format(tuple_shapes))
        else:
            print('x385: {}'.format(x385))
        x386=x385.unsqueeze(0)
        if x386 is None:
            print('x386: {}'.format(x386))
        elif isinstance(x386, torch.Tensor):
            print('x386: {}'.format(x386.shape))
        elif isinstance(x386, tuple):
            tuple_shapes = '('
            for item in x386:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x386: {}'.format(tuple_shapes))
        else:
            print('x386: {}'.format(x386))
        x391=torchvision.models.swin_transformer.shifted_window_attention(x379, self.weight30, self.weight31, x386, [7, 7], 16,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias30, proj_bias=self.bias31)
        if x391 is None:
            print('x391: {}'.format(x391))
        elif isinstance(x391, torch.Tensor):
            print('x391: {}'.format(x391.shape))
        elif isinstance(x391, tuple):
            tuple_shapes = '('
            for item in x391:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x391: {}'.format(tuple_shapes))
        else:
            print('x391: {}'.format(x391))
        x392=stochastic_depth(x391, 0.32608695652173914, 'row', False)
        if x392 is None:
            print('x392: {}'.format(x392))
        elif isinstance(x392, torch.Tensor):
            print('x392: {}'.format(x392.shape))
        elif isinstance(x392, tuple):
            tuple_shapes = '('
            for item in x392:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x392: {}'.format(tuple_shapes))
        else:
            print('x392: {}'.format(x392))
        x393=operator.add(x378, x392)
        if x393 is None:
            print('x393: {}'.format(x393))
        elif isinstance(x393, torch.Tensor):
            print('x393: {}'.format(x393.shape))
        elif isinstance(x393, tuple):
            tuple_shapes = '('
            for item in x393:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x393: {}'.format(tuple_shapes))
        else:
            print('x393: {}'.format(x393))
        x394=self.layernorm34(x393)
        if x394 is None:
            print('x394: {}'.format(x394))
        elif isinstance(x394, torch.Tensor):
            print('x394: {}'.format(x394.shape))
        elif isinstance(x394, tuple):
            tuple_shapes = '('
            for item in x394:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x394: {}'.format(tuple_shapes))
        else:
            print('x394: {}'.format(x394))
        x395=self.linear32(x394)
        if x395 is None:
            print('x395: {}'.format(x395))
        elif isinstance(x395, torch.Tensor):
            print('x395: {}'.format(x395.shape))
        elif isinstance(x395, tuple):
            tuple_shapes = '('
            for item in x395:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x395: {}'.format(tuple_shapes))
        else:
            print('x395: {}'.format(x395))
        x396=self.gelu15(x395)
        if x396 is None:
            print('x396: {}'.format(x396))
        elif isinstance(x396, torch.Tensor):
            print('x396: {}'.format(x396.shape))
        elif isinstance(x396, tuple):
            tuple_shapes = '('
            for item in x396:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x396: {}'.format(tuple_shapes))
        else:
            print('x396: {}'.format(x396))
        x397=self.dropout30(x396)
        if x397 is None:
            print('x397: {}'.format(x397))
        elif isinstance(x397, torch.Tensor):
            print('x397: {}'.format(x397.shape))
        elif isinstance(x397, tuple):
            tuple_shapes = '('
            for item in x397:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x397: {}'.format(tuple_shapes))
        else:
            print('x397: {}'.format(x397))
        x398=self.linear33(x397)
        if x398 is None:
            print('x398: {}'.format(x398))
        elif isinstance(x398, torch.Tensor):
            print('x398: {}'.format(x398.shape))
        elif isinstance(x398, tuple):
            tuple_shapes = '('
            for item in x398:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x398: {}'.format(tuple_shapes))
        else:
            print('x398: {}'.format(x398))
        x399=self.dropout31(x398)
        if x399 is None:
            print('x399: {}'.format(x399))
        elif isinstance(x399, torch.Tensor):
            print('x399: {}'.format(x399.shape))
        elif isinstance(x399, tuple):
            tuple_shapes = '('
            for item in x399:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x399: {}'.format(tuple_shapes))
        else:
            print('x399: {}'.format(x399))
        x400=stochastic_depth(x399, 0.32608695652173914, 'row', False)
        if x400 is None:
            print('x400: {}'.format(x400))
        elif isinstance(x400, torch.Tensor):
            print('x400: {}'.format(x400.shape))
        elif isinstance(x400, tuple):
            tuple_shapes = '('
            for item in x400:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x400: {}'.format(tuple_shapes))
        else:
            print('x400: {}'.format(x400))
        x401=operator.add(x393, x400)
        if x401 is None:
            print('x401: {}'.format(x401))
        elif isinstance(x401, torch.Tensor):
            print('x401: {}'.format(x401.shape))
        elif isinstance(x401, tuple):
            tuple_shapes = '('
            for item in x401:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x401: {}'.format(tuple_shapes))
        else:
            print('x401: {}'.format(x401))
        x402=self.layernorm35(x401)
        if x402 is None:
            print('x402: {}'.format(x402))
        elif isinstance(x402, torch.Tensor):
            print('x402: {}'.format(x402.shape))
        elif isinstance(x402, tuple):
            tuple_shapes = '('
            for item in x402:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x402: {}'.format(tuple_shapes))
        else:
            print('x402: {}'.format(x402))
        x405=operator.getitem(self.relative_position_bias_table16, self.relative_position_index16)
        if x405 is None:
            print('x405: {}'.format(x405))
        elif isinstance(x405, torch.Tensor):
            print('x405: {}'.format(x405.shape))
        elif isinstance(x405, tuple):
            tuple_shapes = '('
            for item in x405:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x405: {}'.format(tuple_shapes))
        else:
            print('x405: {}'.format(x405))
        x406=x405.view(49, 49, -1)
        if x406 is None:
            print('x406: {}'.format(x406))
        elif isinstance(x406, torch.Tensor):
            print('x406: {}'.format(x406.shape))
        elif isinstance(x406, tuple):
            tuple_shapes = '('
            for item in x406:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x406: {}'.format(tuple_shapes))
        else:
            print('x406: {}'.format(x406))
        x407=x406.permute(2, 0, 1)
        if x407 is None:
            print('x407: {}'.format(x407))
        elif isinstance(x407, torch.Tensor):
            print('x407: {}'.format(x407.shape))
        elif isinstance(x407, tuple):
            tuple_shapes = '('
            for item in x407:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x407: {}'.format(tuple_shapes))
        else:
            print('x407: {}'.format(x407))
        x408=x407.contiguous()
        if x408 is None:
            print('x408: {}'.format(x408))
        elif isinstance(x408, torch.Tensor):
            print('x408: {}'.format(x408.shape))
        elif isinstance(x408, tuple):
            tuple_shapes = '('
            for item in x408:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x408: {}'.format(tuple_shapes))
        else:
            print('x408: {}'.format(x408))
        x409=x408.unsqueeze(0)
        if x409 is None:
            print('x409: {}'.format(x409))
        elif isinstance(x409, torch.Tensor):
            print('x409: {}'.format(x409.shape))
        elif isinstance(x409, tuple):
            tuple_shapes = '('
            for item in x409:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x409: {}'.format(tuple_shapes))
        else:
            print('x409: {}'.format(x409))
        x414=torchvision.models.swin_transformer.shifted_window_attention(x402, self.weight32, self.weight33, x409, [7, 7], 16,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias32, proj_bias=self.bias33)
        if x414 is None:
            print('x414: {}'.format(x414))
        elif isinstance(x414, torch.Tensor):
            print('x414: {}'.format(x414.shape))
        elif isinstance(x414, tuple):
            tuple_shapes = '('
            for item in x414:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x414: {}'.format(tuple_shapes))
        else:
            print('x414: {}'.format(x414))
        x415=stochastic_depth(x414, 0.34782608695652173, 'row', False)
        if x415 is None:
            print('x415: {}'.format(x415))
        elif isinstance(x415, torch.Tensor):
            print('x415: {}'.format(x415.shape))
        elif isinstance(x415, tuple):
            tuple_shapes = '('
            for item in x415:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x415: {}'.format(tuple_shapes))
        else:
            print('x415: {}'.format(x415))
        x416=operator.add(x401, x415)
        if x416 is None:
            print('x416: {}'.format(x416))
        elif isinstance(x416, torch.Tensor):
            print('x416: {}'.format(x416.shape))
        elif isinstance(x416, tuple):
            tuple_shapes = '('
            for item in x416:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x416: {}'.format(tuple_shapes))
        else:
            print('x416: {}'.format(x416))
        x417=self.layernorm36(x416)
        if x417 is None:
            print('x417: {}'.format(x417))
        elif isinstance(x417, torch.Tensor):
            print('x417: {}'.format(x417.shape))
        elif isinstance(x417, tuple):
            tuple_shapes = '('
            for item in x417:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x417: {}'.format(tuple_shapes))
        else:
            print('x417: {}'.format(x417))
        x418=self.linear34(x417)
        if x418 is None:
            print('x418: {}'.format(x418))
        elif isinstance(x418, torch.Tensor):
            print('x418: {}'.format(x418.shape))
        elif isinstance(x418, tuple):
            tuple_shapes = '('
            for item in x418:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x418: {}'.format(tuple_shapes))
        else:
            print('x418: {}'.format(x418))
        x419=self.gelu16(x418)
        if x419 is None:
            print('x419: {}'.format(x419))
        elif isinstance(x419, torch.Tensor):
            print('x419: {}'.format(x419.shape))
        elif isinstance(x419, tuple):
            tuple_shapes = '('
            for item in x419:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x419: {}'.format(tuple_shapes))
        else:
            print('x419: {}'.format(x419))
        x420=self.dropout32(x419)
        if x420 is None:
            print('x420: {}'.format(x420))
        elif isinstance(x420, torch.Tensor):
            print('x420: {}'.format(x420.shape))
        elif isinstance(x420, tuple):
            tuple_shapes = '('
            for item in x420:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x420: {}'.format(tuple_shapes))
        else:
            print('x420: {}'.format(x420))
        x421=self.linear35(x420)
        if x421 is None:
            print('x421: {}'.format(x421))
        elif isinstance(x421, torch.Tensor):
            print('x421: {}'.format(x421.shape))
        elif isinstance(x421, tuple):
            tuple_shapes = '('
            for item in x421:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x421: {}'.format(tuple_shapes))
        else:
            print('x421: {}'.format(x421))
        x422=self.dropout33(x421)
        if x422 is None:
            print('x422: {}'.format(x422))
        elif isinstance(x422, torch.Tensor):
            print('x422: {}'.format(x422.shape))
        elif isinstance(x422, tuple):
            tuple_shapes = '('
            for item in x422:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x422: {}'.format(tuple_shapes))
        else:
            print('x422: {}'.format(x422))
        x423=stochastic_depth(x422, 0.34782608695652173, 'row', False)
        if x423 is None:
            print('x423: {}'.format(x423))
        elif isinstance(x423, torch.Tensor):
            print('x423: {}'.format(x423.shape))
        elif isinstance(x423, tuple):
            tuple_shapes = '('
            for item in x423:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x423: {}'.format(tuple_shapes))
        else:
            print('x423: {}'.format(x423))
        x424=operator.add(x416, x423)
        if x424 is None:
            print('x424: {}'.format(x424))
        elif isinstance(x424, torch.Tensor):
            print('x424: {}'.format(x424.shape))
        elif isinstance(x424, tuple):
            tuple_shapes = '('
            for item in x424:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x424: {}'.format(tuple_shapes))
        else:
            print('x424: {}'.format(x424))
        x425=self.layernorm37(x424)
        if x425 is None:
            print('x425: {}'.format(x425))
        elif isinstance(x425, torch.Tensor):
            print('x425: {}'.format(x425.shape))
        elif isinstance(x425, tuple):
            tuple_shapes = '('
            for item in x425:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x425: {}'.format(tuple_shapes))
        else:
            print('x425: {}'.format(x425))
        x428=operator.getitem(self.relative_position_bias_table17, self.relative_position_index17)
        if x428 is None:
            print('x428: {}'.format(x428))
        elif isinstance(x428, torch.Tensor):
            print('x428: {}'.format(x428.shape))
        elif isinstance(x428, tuple):
            tuple_shapes = '('
            for item in x428:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x428: {}'.format(tuple_shapes))
        else:
            print('x428: {}'.format(x428))
        x429=x428.view(49, 49, -1)
        if x429 is None:
            print('x429: {}'.format(x429))
        elif isinstance(x429, torch.Tensor):
            print('x429: {}'.format(x429.shape))
        elif isinstance(x429, tuple):
            tuple_shapes = '('
            for item in x429:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x429: {}'.format(tuple_shapes))
        else:
            print('x429: {}'.format(x429))
        x430=x429.permute(2, 0, 1)
        if x430 is None:
            print('x430: {}'.format(x430))
        elif isinstance(x430, torch.Tensor):
            print('x430: {}'.format(x430.shape))
        elif isinstance(x430, tuple):
            tuple_shapes = '('
            for item in x430:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x430: {}'.format(tuple_shapes))
        else:
            print('x430: {}'.format(x430))
        x431=x430.contiguous()
        if x431 is None:
            print('x431: {}'.format(x431))
        elif isinstance(x431, torch.Tensor):
            print('x431: {}'.format(x431.shape))
        elif isinstance(x431, tuple):
            tuple_shapes = '('
            for item in x431:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x431: {}'.format(tuple_shapes))
        else:
            print('x431: {}'.format(x431))
        x432=x431.unsqueeze(0)
        if x432 is None:
            print('x432: {}'.format(x432))
        elif isinstance(x432, torch.Tensor):
            print('x432: {}'.format(x432.shape))
        elif isinstance(x432, tuple):
            tuple_shapes = '('
            for item in x432:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x432: {}'.format(tuple_shapes))
        else:
            print('x432: {}'.format(x432))
        x437=torchvision.models.swin_transformer.shifted_window_attention(x425, self.weight34, self.weight35, x432, [7, 7], 16,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias34, proj_bias=self.bias35)
        if x437 is None:
            print('x437: {}'.format(x437))
        elif isinstance(x437, torch.Tensor):
            print('x437: {}'.format(x437.shape))
        elif isinstance(x437, tuple):
            tuple_shapes = '('
            for item in x437:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x437: {}'.format(tuple_shapes))
        else:
            print('x437: {}'.format(x437))
        x438=stochastic_depth(x437, 0.3695652173913043, 'row', False)
        if x438 is None:
            print('x438: {}'.format(x438))
        elif isinstance(x438, torch.Tensor):
            print('x438: {}'.format(x438.shape))
        elif isinstance(x438, tuple):
            tuple_shapes = '('
            for item in x438:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x438: {}'.format(tuple_shapes))
        else:
            print('x438: {}'.format(x438))
        x439=operator.add(x424, x438)
        if x439 is None:
            print('x439: {}'.format(x439))
        elif isinstance(x439, torch.Tensor):
            print('x439: {}'.format(x439.shape))
        elif isinstance(x439, tuple):
            tuple_shapes = '('
            for item in x439:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x439: {}'.format(tuple_shapes))
        else:
            print('x439: {}'.format(x439))
        x440=self.layernorm38(x439)
        if x440 is None:
            print('x440: {}'.format(x440))
        elif isinstance(x440, torch.Tensor):
            print('x440: {}'.format(x440.shape))
        elif isinstance(x440, tuple):
            tuple_shapes = '('
            for item in x440:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x440: {}'.format(tuple_shapes))
        else:
            print('x440: {}'.format(x440))
        x441=self.linear36(x440)
        if x441 is None:
            print('x441: {}'.format(x441))
        elif isinstance(x441, torch.Tensor):
            print('x441: {}'.format(x441.shape))
        elif isinstance(x441, tuple):
            tuple_shapes = '('
            for item in x441:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x441: {}'.format(tuple_shapes))
        else:
            print('x441: {}'.format(x441))
        x442=self.gelu17(x441)
        if x442 is None:
            print('x442: {}'.format(x442))
        elif isinstance(x442, torch.Tensor):
            print('x442: {}'.format(x442.shape))
        elif isinstance(x442, tuple):
            tuple_shapes = '('
            for item in x442:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x442: {}'.format(tuple_shapes))
        else:
            print('x442: {}'.format(x442))
        x443=self.dropout34(x442)
        if x443 is None:
            print('x443: {}'.format(x443))
        elif isinstance(x443, torch.Tensor):
            print('x443: {}'.format(x443.shape))
        elif isinstance(x443, tuple):
            tuple_shapes = '('
            for item in x443:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x443: {}'.format(tuple_shapes))
        else:
            print('x443: {}'.format(x443))
        x444=self.linear37(x443)
        if x444 is None:
            print('x444: {}'.format(x444))
        elif isinstance(x444, torch.Tensor):
            print('x444: {}'.format(x444.shape))
        elif isinstance(x444, tuple):
            tuple_shapes = '('
            for item in x444:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x444: {}'.format(tuple_shapes))
        else:
            print('x444: {}'.format(x444))
        x445=self.dropout35(x444)
        if x445 is None:
            print('x445: {}'.format(x445))
        elif isinstance(x445, torch.Tensor):
            print('x445: {}'.format(x445.shape))
        elif isinstance(x445, tuple):
            tuple_shapes = '('
            for item in x445:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x445: {}'.format(tuple_shapes))
        else:
            print('x445: {}'.format(x445))
        x446=stochastic_depth(x445, 0.3695652173913043, 'row', False)
        if x446 is None:
            print('x446: {}'.format(x446))
        elif isinstance(x446, torch.Tensor):
            print('x446: {}'.format(x446.shape))
        elif isinstance(x446, tuple):
            tuple_shapes = '('
            for item in x446:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x446: {}'.format(tuple_shapes))
        else:
            print('x446: {}'.format(x446))
        x447=operator.add(x439, x446)
        if x447 is None:
            print('x447: {}'.format(x447))
        elif isinstance(x447, torch.Tensor):
            print('x447: {}'.format(x447.shape))
        elif isinstance(x447, tuple):
            tuple_shapes = '('
            for item in x447:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x447: {}'.format(tuple_shapes))
        else:
            print('x447: {}'.format(x447))
        x448=self.layernorm39(x447)
        if x448 is None:
            print('x448: {}'.format(x448))
        elif isinstance(x448, torch.Tensor):
            print('x448: {}'.format(x448.shape))
        elif isinstance(x448, tuple):
            tuple_shapes = '('
            for item in x448:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x448: {}'.format(tuple_shapes))
        else:
            print('x448: {}'.format(x448))
        x451=operator.getitem(self.relative_position_bias_table18, self.relative_position_index18)
        if x451 is None:
            print('x451: {}'.format(x451))
        elif isinstance(x451, torch.Tensor):
            print('x451: {}'.format(x451.shape))
        elif isinstance(x451, tuple):
            tuple_shapes = '('
            for item in x451:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x451: {}'.format(tuple_shapes))
        else:
            print('x451: {}'.format(x451))
        x452=x451.view(49, 49, -1)
        if x452 is None:
            print('x452: {}'.format(x452))
        elif isinstance(x452, torch.Tensor):
            print('x452: {}'.format(x452.shape))
        elif isinstance(x452, tuple):
            tuple_shapes = '('
            for item in x452:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x452: {}'.format(tuple_shapes))
        else:
            print('x452: {}'.format(x452))
        x453=x452.permute(2, 0, 1)
        if x453 is None:
            print('x453: {}'.format(x453))
        elif isinstance(x453, torch.Tensor):
            print('x453: {}'.format(x453.shape))
        elif isinstance(x453, tuple):
            tuple_shapes = '('
            for item in x453:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x453: {}'.format(tuple_shapes))
        else:
            print('x453: {}'.format(x453))
        x454=x453.contiguous()
        if x454 is None:
            print('x454: {}'.format(x454))
        elif isinstance(x454, torch.Tensor):
            print('x454: {}'.format(x454.shape))
        elif isinstance(x454, tuple):
            tuple_shapes = '('
            for item in x454:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x454: {}'.format(tuple_shapes))
        else:
            print('x454: {}'.format(x454))
        x455=x454.unsqueeze(0)
        if x455 is None:
            print('x455: {}'.format(x455))
        elif isinstance(x455, torch.Tensor):
            print('x455: {}'.format(x455.shape))
        elif isinstance(x455, tuple):
            tuple_shapes = '('
            for item in x455:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x455: {}'.format(tuple_shapes))
        else:
            print('x455: {}'.format(x455))
        x460=torchvision.models.swin_transformer.shifted_window_attention(x448, self.weight36, self.weight37, x455, [7, 7], 16,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias36, proj_bias=self.bias37)
        if x460 is None:
            print('x460: {}'.format(x460))
        elif isinstance(x460, torch.Tensor):
            print('x460: {}'.format(x460.shape))
        elif isinstance(x460, tuple):
            tuple_shapes = '('
            for item in x460:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x460: {}'.format(tuple_shapes))
        else:
            print('x460: {}'.format(x460))
        x461=stochastic_depth(x460, 0.391304347826087, 'row', False)
        if x461 is None:
            print('x461: {}'.format(x461))
        elif isinstance(x461, torch.Tensor):
            print('x461: {}'.format(x461.shape))
        elif isinstance(x461, tuple):
            tuple_shapes = '('
            for item in x461:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x461: {}'.format(tuple_shapes))
        else:
            print('x461: {}'.format(x461))
        x462=operator.add(x447, x461)
        if x462 is None:
            print('x462: {}'.format(x462))
        elif isinstance(x462, torch.Tensor):
            print('x462: {}'.format(x462.shape))
        elif isinstance(x462, tuple):
            tuple_shapes = '('
            for item in x462:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x462: {}'.format(tuple_shapes))
        else:
            print('x462: {}'.format(x462))
        x463=self.layernorm40(x462)
        if x463 is None:
            print('x463: {}'.format(x463))
        elif isinstance(x463, torch.Tensor):
            print('x463: {}'.format(x463.shape))
        elif isinstance(x463, tuple):
            tuple_shapes = '('
            for item in x463:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x463: {}'.format(tuple_shapes))
        else:
            print('x463: {}'.format(x463))
        x464=self.linear38(x463)
        if x464 is None:
            print('x464: {}'.format(x464))
        elif isinstance(x464, torch.Tensor):
            print('x464: {}'.format(x464.shape))
        elif isinstance(x464, tuple):
            tuple_shapes = '('
            for item in x464:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x464: {}'.format(tuple_shapes))
        else:
            print('x464: {}'.format(x464))
        x465=self.gelu18(x464)
        if x465 is None:
            print('x465: {}'.format(x465))
        elif isinstance(x465, torch.Tensor):
            print('x465: {}'.format(x465.shape))
        elif isinstance(x465, tuple):
            tuple_shapes = '('
            for item in x465:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x465: {}'.format(tuple_shapes))
        else:
            print('x465: {}'.format(x465))
        x466=self.dropout36(x465)
        if x466 is None:
            print('x466: {}'.format(x466))
        elif isinstance(x466, torch.Tensor):
            print('x466: {}'.format(x466.shape))
        elif isinstance(x466, tuple):
            tuple_shapes = '('
            for item in x466:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x466: {}'.format(tuple_shapes))
        else:
            print('x466: {}'.format(x466))
        x467=self.linear39(x466)
        if x467 is None:
            print('x467: {}'.format(x467))
        elif isinstance(x467, torch.Tensor):
            print('x467: {}'.format(x467.shape))
        elif isinstance(x467, tuple):
            tuple_shapes = '('
            for item in x467:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x467: {}'.format(tuple_shapes))
        else:
            print('x467: {}'.format(x467))
        x468=self.dropout37(x467)
        if x468 is None:
            print('x468: {}'.format(x468))
        elif isinstance(x468, torch.Tensor):
            print('x468: {}'.format(x468.shape))
        elif isinstance(x468, tuple):
            tuple_shapes = '('
            for item in x468:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x468: {}'.format(tuple_shapes))
        else:
            print('x468: {}'.format(x468))
        x469=stochastic_depth(x468, 0.391304347826087, 'row', False)
        if x469 is None:
            print('x469: {}'.format(x469))
        elif isinstance(x469, torch.Tensor):
            print('x469: {}'.format(x469.shape))
        elif isinstance(x469, tuple):
            tuple_shapes = '('
            for item in x469:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x469: {}'.format(tuple_shapes))
        else:
            print('x469: {}'.format(x469))
        x470=operator.add(x462, x469)
        if x470 is None:
            print('x470: {}'.format(x470))
        elif isinstance(x470, torch.Tensor):
            print('x470: {}'.format(x470.shape))
        elif isinstance(x470, tuple):
            tuple_shapes = '('
            for item in x470:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x470: {}'.format(tuple_shapes))
        else:
            print('x470: {}'.format(x470))
        x471=self.layernorm41(x470)
        if x471 is None:
            print('x471: {}'.format(x471))
        elif isinstance(x471, torch.Tensor):
            print('x471: {}'.format(x471.shape))
        elif isinstance(x471, tuple):
            tuple_shapes = '('
            for item in x471:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x471: {}'.format(tuple_shapes))
        else:
            print('x471: {}'.format(x471))
        x474=operator.getitem(self.relative_position_bias_table19, self.relative_position_index19)
        if x474 is None:
            print('x474: {}'.format(x474))
        elif isinstance(x474, torch.Tensor):
            print('x474: {}'.format(x474.shape))
        elif isinstance(x474, tuple):
            tuple_shapes = '('
            for item in x474:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x474: {}'.format(tuple_shapes))
        else:
            print('x474: {}'.format(x474))
        x475=x474.view(49, 49, -1)
        if x475 is None:
            print('x475: {}'.format(x475))
        elif isinstance(x475, torch.Tensor):
            print('x475: {}'.format(x475.shape))
        elif isinstance(x475, tuple):
            tuple_shapes = '('
            for item in x475:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x475: {}'.format(tuple_shapes))
        else:
            print('x475: {}'.format(x475))
        x476=x475.permute(2, 0, 1)
        if x476 is None:
            print('x476: {}'.format(x476))
        elif isinstance(x476, torch.Tensor):
            print('x476: {}'.format(x476.shape))
        elif isinstance(x476, tuple):
            tuple_shapes = '('
            for item in x476:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x476: {}'.format(tuple_shapes))
        else:
            print('x476: {}'.format(x476))
        x477=x476.contiguous()
        if x477 is None:
            print('x477: {}'.format(x477))
        elif isinstance(x477, torch.Tensor):
            print('x477: {}'.format(x477.shape))
        elif isinstance(x477, tuple):
            tuple_shapes = '('
            for item in x477:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x477: {}'.format(tuple_shapes))
        else:
            print('x477: {}'.format(x477))
        x478=x477.unsqueeze(0)
        if x478 is None:
            print('x478: {}'.format(x478))
        elif isinstance(x478, torch.Tensor):
            print('x478: {}'.format(x478.shape))
        elif isinstance(x478, tuple):
            tuple_shapes = '('
            for item in x478:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x478: {}'.format(tuple_shapes))
        else:
            print('x478: {}'.format(x478))
        x483=torchvision.models.swin_transformer.shifted_window_attention(x471, self.weight38, self.weight39, x478, [7, 7], 16,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias38, proj_bias=self.bias39)
        if x483 is None:
            print('x483: {}'.format(x483))
        elif isinstance(x483, torch.Tensor):
            print('x483: {}'.format(x483.shape))
        elif isinstance(x483, tuple):
            tuple_shapes = '('
            for item in x483:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x483: {}'.format(tuple_shapes))
        else:
            print('x483: {}'.format(x483))
        x484=stochastic_depth(x483, 0.41304347826086957, 'row', False)
        if x484 is None:
            print('x484: {}'.format(x484))
        elif isinstance(x484, torch.Tensor):
            print('x484: {}'.format(x484.shape))
        elif isinstance(x484, tuple):
            tuple_shapes = '('
            for item in x484:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x484: {}'.format(tuple_shapes))
        else:
            print('x484: {}'.format(x484))
        x485=operator.add(x470, x484)
        if x485 is None:
            print('x485: {}'.format(x485))
        elif isinstance(x485, torch.Tensor):
            print('x485: {}'.format(x485.shape))
        elif isinstance(x485, tuple):
            tuple_shapes = '('
            for item in x485:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x485: {}'.format(tuple_shapes))
        else:
            print('x485: {}'.format(x485))
        x486=self.layernorm42(x485)
        if x486 is None:
            print('x486: {}'.format(x486))
        elif isinstance(x486, torch.Tensor):
            print('x486: {}'.format(x486.shape))
        elif isinstance(x486, tuple):
            tuple_shapes = '('
            for item in x486:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x486: {}'.format(tuple_shapes))
        else:
            print('x486: {}'.format(x486))
        x487=self.linear40(x486)
        if x487 is None:
            print('x487: {}'.format(x487))
        elif isinstance(x487, torch.Tensor):
            print('x487: {}'.format(x487.shape))
        elif isinstance(x487, tuple):
            tuple_shapes = '('
            for item in x487:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x487: {}'.format(tuple_shapes))
        else:
            print('x487: {}'.format(x487))
        x488=self.gelu19(x487)
        if x488 is None:
            print('x488: {}'.format(x488))
        elif isinstance(x488, torch.Tensor):
            print('x488: {}'.format(x488.shape))
        elif isinstance(x488, tuple):
            tuple_shapes = '('
            for item in x488:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x488: {}'.format(tuple_shapes))
        else:
            print('x488: {}'.format(x488))
        x489=self.dropout38(x488)
        if x489 is None:
            print('x489: {}'.format(x489))
        elif isinstance(x489, torch.Tensor):
            print('x489: {}'.format(x489.shape))
        elif isinstance(x489, tuple):
            tuple_shapes = '('
            for item in x489:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x489: {}'.format(tuple_shapes))
        else:
            print('x489: {}'.format(x489))
        x490=self.linear41(x489)
        if x490 is None:
            print('x490: {}'.format(x490))
        elif isinstance(x490, torch.Tensor):
            print('x490: {}'.format(x490.shape))
        elif isinstance(x490, tuple):
            tuple_shapes = '('
            for item in x490:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x490: {}'.format(tuple_shapes))
        else:
            print('x490: {}'.format(x490))
        x491=self.dropout39(x490)
        if x491 is None:
            print('x491: {}'.format(x491))
        elif isinstance(x491, torch.Tensor):
            print('x491: {}'.format(x491.shape))
        elif isinstance(x491, tuple):
            tuple_shapes = '('
            for item in x491:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x491: {}'.format(tuple_shapes))
        else:
            print('x491: {}'.format(x491))
        x492=stochastic_depth(x491, 0.41304347826086957, 'row', False)
        if x492 is None:
            print('x492: {}'.format(x492))
        elif isinstance(x492, torch.Tensor):
            print('x492: {}'.format(x492.shape))
        elif isinstance(x492, tuple):
            tuple_shapes = '('
            for item in x492:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x492: {}'.format(tuple_shapes))
        else:
            print('x492: {}'.format(x492))
        x493=operator.add(x485, x492)
        if x493 is None:
            print('x493: {}'.format(x493))
        elif isinstance(x493, torch.Tensor):
            print('x493: {}'.format(x493.shape))
        elif isinstance(x493, tuple):
            tuple_shapes = '('
            for item in x493:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x493: {}'.format(tuple_shapes))
        else:
            print('x493: {}'.format(x493))
        x494=self.layernorm43(x493)
        if x494 is None:
            print('x494: {}'.format(x494))
        elif isinstance(x494, torch.Tensor):
            print('x494: {}'.format(x494.shape))
        elif isinstance(x494, tuple):
            tuple_shapes = '('
            for item in x494:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x494: {}'.format(tuple_shapes))
        else:
            print('x494: {}'.format(x494))
        x497=operator.getitem(self.relative_position_bias_table20, self.relative_position_index20)
        if x497 is None:
            print('x497: {}'.format(x497))
        elif isinstance(x497, torch.Tensor):
            print('x497: {}'.format(x497.shape))
        elif isinstance(x497, tuple):
            tuple_shapes = '('
            for item in x497:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x497: {}'.format(tuple_shapes))
        else:
            print('x497: {}'.format(x497))
        x498=x497.view(49, 49, -1)
        if x498 is None:
            print('x498: {}'.format(x498))
        elif isinstance(x498, torch.Tensor):
            print('x498: {}'.format(x498.shape))
        elif isinstance(x498, tuple):
            tuple_shapes = '('
            for item in x498:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x498: {}'.format(tuple_shapes))
        else:
            print('x498: {}'.format(x498))
        x499=x498.permute(2, 0, 1)
        if x499 is None:
            print('x499: {}'.format(x499))
        elif isinstance(x499, torch.Tensor):
            print('x499: {}'.format(x499.shape))
        elif isinstance(x499, tuple):
            tuple_shapes = '('
            for item in x499:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x499: {}'.format(tuple_shapes))
        else:
            print('x499: {}'.format(x499))
        x500=x499.contiguous()
        if x500 is None:
            print('x500: {}'.format(x500))
        elif isinstance(x500, torch.Tensor):
            print('x500: {}'.format(x500.shape))
        elif isinstance(x500, tuple):
            tuple_shapes = '('
            for item in x500:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x500: {}'.format(tuple_shapes))
        else:
            print('x500: {}'.format(x500))
        x501=x500.unsqueeze(0)
        if x501 is None:
            print('x501: {}'.format(x501))
        elif isinstance(x501, torch.Tensor):
            print('x501: {}'.format(x501.shape))
        elif isinstance(x501, tuple):
            tuple_shapes = '('
            for item in x501:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x501: {}'.format(tuple_shapes))
        else:
            print('x501: {}'.format(x501))
        x506=torchvision.models.swin_transformer.shifted_window_attention(x494, self.weight40, self.weight41, x501, [7, 7], 16,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias40, proj_bias=self.bias41)
        if x506 is None:
            print('x506: {}'.format(x506))
        elif isinstance(x506, torch.Tensor):
            print('x506: {}'.format(x506.shape))
        elif isinstance(x506, tuple):
            tuple_shapes = '('
            for item in x506:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x506: {}'.format(tuple_shapes))
        else:
            print('x506: {}'.format(x506))
        x507=stochastic_depth(x506, 0.43478260869565216, 'row', False)
        if x507 is None:
            print('x507: {}'.format(x507))
        elif isinstance(x507, torch.Tensor):
            print('x507: {}'.format(x507.shape))
        elif isinstance(x507, tuple):
            tuple_shapes = '('
            for item in x507:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x507: {}'.format(tuple_shapes))
        else:
            print('x507: {}'.format(x507))
        x508=operator.add(x493, x507)
        if x508 is None:
            print('x508: {}'.format(x508))
        elif isinstance(x508, torch.Tensor):
            print('x508: {}'.format(x508.shape))
        elif isinstance(x508, tuple):
            tuple_shapes = '('
            for item in x508:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x508: {}'.format(tuple_shapes))
        else:
            print('x508: {}'.format(x508))
        x509=self.layernorm44(x508)
        if x509 is None:
            print('x509: {}'.format(x509))
        elif isinstance(x509, torch.Tensor):
            print('x509: {}'.format(x509.shape))
        elif isinstance(x509, tuple):
            tuple_shapes = '('
            for item in x509:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x509: {}'.format(tuple_shapes))
        else:
            print('x509: {}'.format(x509))
        x510=self.linear42(x509)
        if x510 is None:
            print('x510: {}'.format(x510))
        elif isinstance(x510, torch.Tensor):
            print('x510: {}'.format(x510.shape))
        elif isinstance(x510, tuple):
            tuple_shapes = '('
            for item in x510:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x510: {}'.format(tuple_shapes))
        else:
            print('x510: {}'.format(x510))
        x511=self.gelu20(x510)
        if x511 is None:
            print('x511: {}'.format(x511))
        elif isinstance(x511, torch.Tensor):
            print('x511: {}'.format(x511.shape))
        elif isinstance(x511, tuple):
            tuple_shapes = '('
            for item in x511:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x511: {}'.format(tuple_shapes))
        else:
            print('x511: {}'.format(x511))
        x512=self.dropout40(x511)
        if x512 is None:
            print('x512: {}'.format(x512))
        elif isinstance(x512, torch.Tensor):
            print('x512: {}'.format(x512.shape))
        elif isinstance(x512, tuple):
            tuple_shapes = '('
            for item in x512:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x512: {}'.format(tuple_shapes))
        else:
            print('x512: {}'.format(x512))
        x513=self.linear43(x512)
        if x513 is None:
            print('x513: {}'.format(x513))
        elif isinstance(x513, torch.Tensor):
            print('x513: {}'.format(x513.shape))
        elif isinstance(x513, tuple):
            tuple_shapes = '('
            for item in x513:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x513: {}'.format(tuple_shapes))
        else:
            print('x513: {}'.format(x513))
        x514=self.dropout41(x513)
        if x514 is None:
            print('x514: {}'.format(x514))
        elif isinstance(x514, torch.Tensor):
            print('x514: {}'.format(x514.shape))
        elif isinstance(x514, tuple):
            tuple_shapes = '('
            for item in x514:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x514: {}'.format(tuple_shapes))
        else:
            print('x514: {}'.format(x514))
        x515=stochastic_depth(x514, 0.43478260869565216, 'row', False)
        if x515 is None:
            print('x515: {}'.format(x515))
        elif isinstance(x515, torch.Tensor):
            print('x515: {}'.format(x515.shape))
        elif isinstance(x515, tuple):
            tuple_shapes = '('
            for item in x515:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x515: {}'.format(tuple_shapes))
        else:
            print('x515: {}'.format(x515))
        x516=operator.add(x508, x515)
        if x516 is None:
            print('x516: {}'.format(x516))
        elif isinstance(x516, torch.Tensor):
            print('x516: {}'.format(x516.shape))
        elif isinstance(x516, tuple):
            tuple_shapes = '('
            for item in x516:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x516: {}'.format(tuple_shapes))
        else:
            print('x516: {}'.format(x516))
        x517=self.layernorm45(x516)
        if x517 is None:
            print('x517: {}'.format(x517))
        elif isinstance(x517, torch.Tensor):
            print('x517: {}'.format(x517.shape))
        elif isinstance(x517, tuple):
            tuple_shapes = '('
            for item in x517:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x517: {}'.format(tuple_shapes))
        else:
            print('x517: {}'.format(x517))
        x520=operator.getitem(self.relative_position_bias_table21, self.relative_position_index21)
        if x520 is None:
            print('x520: {}'.format(x520))
        elif isinstance(x520, torch.Tensor):
            print('x520: {}'.format(x520.shape))
        elif isinstance(x520, tuple):
            tuple_shapes = '('
            for item in x520:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x520: {}'.format(tuple_shapes))
        else:
            print('x520: {}'.format(x520))
        x521=x520.view(49, 49, -1)
        if x521 is None:
            print('x521: {}'.format(x521))
        elif isinstance(x521, torch.Tensor):
            print('x521: {}'.format(x521.shape))
        elif isinstance(x521, tuple):
            tuple_shapes = '('
            for item in x521:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x521: {}'.format(tuple_shapes))
        else:
            print('x521: {}'.format(x521))
        x522=x521.permute(2, 0, 1)
        if x522 is None:
            print('x522: {}'.format(x522))
        elif isinstance(x522, torch.Tensor):
            print('x522: {}'.format(x522.shape))
        elif isinstance(x522, tuple):
            tuple_shapes = '('
            for item in x522:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x522: {}'.format(tuple_shapes))
        else:
            print('x522: {}'.format(x522))
        x523=x522.contiguous()
        if x523 is None:
            print('x523: {}'.format(x523))
        elif isinstance(x523, torch.Tensor):
            print('x523: {}'.format(x523.shape))
        elif isinstance(x523, tuple):
            tuple_shapes = '('
            for item in x523:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x523: {}'.format(tuple_shapes))
        else:
            print('x523: {}'.format(x523))
        x524=x523.unsqueeze(0)
        if x524 is None:
            print('x524: {}'.format(x524))
        elif isinstance(x524, torch.Tensor):
            print('x524: {}'.format(x524.shape))
        elif isinstance(x524, tuple):
            tuple_shapes = '('
            for item in x524:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x524: {}'.format(tuple_shapes))
        else:
            print('x524: {}'.format(x524))
        x529=torchvision.models.swin_transformer.shifted_window_attention(x517, self.weight42, self.weight43, x524, [7, 7], 16,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias42, proj_bias=self.bias43)
        if x529 is None:
            print('x529: {}'.format(x529))
        elif isinstance(x529, torch.Tensor):
            print('x529: {}'.format(x529.shape))
        elif isinstance(x529, tuple):
            tuple_shapes = '('
            for item in x529:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x529: {}'.format(tuple_shapes))
        else:
            print('x529: {}'.format(x529))
        x530=stochastic_depth(x529, 0.45652173913043476, 'row', False)
        if x530 is None:
            print('x530: {}'.format(x530))
        elif isinstance(x530, torch.Tensor):
            print('x530: {}'.format(x530.shape))
        elif isinstance(x530, tuple):
            tuple_shapes = '('
            for item in x530:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x530: {}'.format(tuple_shapes))
        else:
            print('x530: {}'.format(x530))
        x531=operator.add(x516, x530)
        if x531 is None:
            print('x531: {}'.format(x531))
        elif isinstance(x531, torch.Tensor):
            print('x531: {}'.format(x531.shape))
        elif isinstance(x531, tuple):
            tuple_shapes = '('
            for item in x531:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x531: {}'.format(tuple_shapes))
        else:
            print('x531: {}'.format(x531))
        x532=self.layernorm46(x531)
        if x532 is None:
            print('x532: {}'.format(x532))
        elif isinstance(x532, torch.Tensor):
            print('x532: {}'.format(x532.shape))
        elif isinstance(x532, tuple):
            tuple_shapes = '('
            for item in x532:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x532: {}'.format(tuple_shapes))
        else:
            print('x532: {}'.format(x532))
        x533=self.linear44(x532)
        if x533 is None:
            print('x533: {}'.format(x533))
        elif isinstance(x533, torch.Tensor):
            print('x533: {}'.format(x533.shape))
        elif isinstance(x533, tuple):
            tuple_shapes = '('
            for item in x533:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x533: {}'.format(tuple_shapes))
        else:
            print('x533: {}'.format(x533))
        x534=self.gelu21(x533)
        if x534 is None:
            print('x534: {}'.format(x534))
        elif isinstance(x534, torch.Tensor):
            print('x534: {}'.format(x534.shape))
        elif isinstance(x534, tuple):
            tuple_shapes = '('
            for item in x534:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x534: {}'.format(tuple_shapes))
        else:
            print('x534: {}'.format(x534))
        x535=self.dropout42(x534)
        if x535 is None:
            print('x535: {}'.format(x535))
        elif isinstance(x535, torch.Tensor):
            print('x535: {}'.format(x535.shape))
        elif isinstance(x535, tuple):
            tuple_shapes = '('
            for item in x535:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x535: {}'.format(tuple_shapes))
        else:
            print('x535: {}'.format(x535))
        x536=self.linear45(x535)
        if x536 is None:
            print('x536: {}'.format(x536))
        elif isinstance(x536, torch.Tensor):
            print('x536: {}'.format(x536.shape))
        elif isinstance(x536, tuple):
            tuple_shapes = '('
            for item in x536:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x536: {}'.format(tuple_shapes))
        else:
            print('x536: {}'.format(x536))
        x537=self.dropout43(x536)
        if x537 is None:
            print('x537: {}'.format(x537))
        elif isinstance(x537, torch.Tensor):
            print('x537: {}'.format(x537.shape))
        elif isinstance(x537, tuple):
            tuple_shapes = '('
            for item in x537:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x537: {}'.format(tuple_shapes))
        else:
            print('x537: {}'.format(x537))
        x538=stochastic_depth(x537, 0.45652173913043476, 'row', False)
        if x538 is None:
            print('x538: {}'.format(x538))
        elif isinstance(x538, torch.Tensor):
            print('x538: {}'.format(x538.shape))
        elif isinstance(x538, tuple):
            tuple_shapes = '('
            for item in x538:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x538: {}'.format(tuple_shapes))
        else:
            print('x538: {}'.format(x538))
        x539=operator.add(x531, x538)
        if x539 is None:
            print('x539: {}'.format(x539))
        elif isinstance(x539, torch.Tensor):
            print('x539: {}'.format(x539.shape))
        elif isinstance(x539, tuple):
            tuple_shapes = '('
            for item in x539:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x539: {}'.format(tuple_shapes))
        else:
            print('x539: {}'.format(x539))
        x540=builtins.getattr(x539, 'shape')
        if x540 is None:
            print('x540: {}'.format(x540))
        elif isinstance(x540, torch.Tensor):
            print('x540: {}'.format(x540.shape))
        elif isinstance(x540, tuple):
            tuple_shapes = '('
            for item in x540:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x540: {}'.format(tuple_shapes))
        else:
            print('x540: {}'.format(x540))
        x541=operator.getitem(x540, slice(-3, None, None))
        if x541 is None:
            print('x541: {}'.format(x541))
        elif isinstance(x541, torch.Tensor):
            print('x541: {}'.format(x541.shape))
        elif isinstance(x541, tuple):
            tuple_shapes = '('
            for item in x541:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x541: {}'.format(tuple_shapes))
        else:
            print('x541: {}'.format(x541))
        x542=operator.getitem(x541, 0)
        if x542 is None:
            print('x542: {}'.format(x542))
        elif isinstance(x542, torch.Tensor):
            print('x542: {}'.format(x542.shape))
        elif isinstance(x542, tuple):
            tuple_shapes = '('
            for item in x542:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x542: {}'.format(tuple_shapes))
        else:
            print('x542: {}'.format(x542))
        x543=operator.getitem(x541, 1)
        if x543 is None:
            print('x543: {}'.format(x543))
        elif isinstance(x543, torch.Tensor):
            print('x543: {}'.format(x543.shape))
        elif isinstance(x543, tuple):
            tuple_shapes = '('
            for item in x543:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x543: {}'.format(tuple_shapes))
        else:
            print('x543: {}'.format(x543))
        x544=operator.getitem(x541, 2)
        if x544 is None:
            print('x544: {}'.format(x544))
        elif isinstance(x544, torch.Tensor):
            print('x544: {}'.format(x544.shape))
        elif isinstance(x544, tuple):
            tuple_shapes = '('
            for item in x544:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x544: {}'.format(tuple_shapes))
        else:
            print('x544: {}'.format(x544))
        x545=operator.mod(x543, 2)
        if x545 is None:
            print('x545: {}'.format(x545))
        elif isinstance(x545, torch.Tensor):
            print('x545: {}'.format(x545.shape))
        elif isinstance(x545, tuple):
            tuple_shapes = '('
            for item in x545:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x545: {}'.format(tuple_shapes))
        else:
            print('x545: {}'.format(x545))
        x546=operator.mod(x542, 2)
        if x546 is None:
            print('x546: {}'.format(x546))
        elif isinstance(x546, torch.Tensor):
            print('x546: {}'.format(x546.shape))
        elif isinstance(x546, tuple):
            tuple_shapes = '('
            for item in x546:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x546: {}'.format(tuple_shapes))
        else:
            print('x546: {}'.format(x546))
        x547=torch.nn.functional.pad(x539, (0, 0, 0, x545, 0, x546))
        if x547 is None:
            print('x547: {}'.format(x547))
        elif isinstance(x547, torch.Tensor):
            print('x547: {}'.format(x547.shape))
        elif isinstance(x547, tuple):
            tuple_shapes = '('
            for item in x547:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x547: {}'.format(tuple_shapes))
        else:
            print('x547: {}'.format(x547))
        x548=operator.getitem(x547, (Ellipsis, slice(0, None, 2), slice(0, None, 2), slice(None, None, None)))
        if x548 is None:
            print('x548: {}'.format(x548))
        elif isinstance(x548, torch.Tensor):
            print('x548: {}'.format(x548.shape))
        elif isinstance(x548, tuple):
            tuple_shapes = '('
            for item in x548:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x548: {}'.format(tuple_shapes))
        else:
            print('x548: {}'.format(x548))
        x549=operator.getitem(x547, (Ellipsis, slice(1, None, 2), slice(0, None, 2), slice(None, None, None)))
        if x549 is None:
            print('x549: {}'.format(x549))
        elif isinstance(x549, torch.Tensor):
            print('x549: {}'.format(x549.shape))
        elif isinstance(x549, tuple):
            tuple_shapes = '('
            for item in x549:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x549: {}'.format(tuple_shapes))
        else:
            print('x549: {}'.format(x549))
        x550=operator.getitem(x547, (Ellipsis, slice(0, None, 2), slice(1, None, 2), slice(None, None, None)))
        if x550 is None:
            print('x550: {}'.format(x550))
        elif isinstance(x550, torch.Tensor):
            print('x550: {}'.format(x550.shape))
        elif isinstance(x550, tuple):
            tuple_shapes = '('
            for item in x550:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x550: {}'.format(tuple_shapes))
        else:
            print('x550: {}'.format(x550))
        x551=operator.getitem(x547, (Ellipsis, slice(1, None, 2), slice(1, None, 2), slice(None, None, None)))
        if x551 is None:
            print('x551: {}'.format(x551))
        elif isinstance(x551, torch.Tensor):
            print('x551: {}'.format(x551.shape))
        elif isinstance(x551, tuple):
            tuple_shapes = '('
            for item in x551:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x551: {}'.format(tuple_shapes))
        else:
            print('x551: {}'.format(x551))
        x552=torch.cat([x548, x549, x550, x551], -1)
        if x552 is None:
            print('x552: {}'.format(x552))
        elif isinstance(x552, torch.Tensor):
            print('x552: {}'.format(x552.shape))
        elif isinstance(x552, tuple):
            tuple_shapes = '('
            for item in x552:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x552: {}'.format(tuple_shapes))
        else:
            print('x552: {}'.format(x552))
        x553=self.layernorm47(x552)
        if x553 is None:
            print('x553: {}'.format(x553))
        elif isinstance(x553, torch.Tensor):
            print('x553: {}'.format(x553.shape))
        elif isinstance(x553, tuple):
            tuple_shapes = '('
            for item in x553:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x553: {}'.format(tuple_shapes))
        else:
            print('x553: {}'.format(x553))
        x554=self.linear46(x553)
        if x554 is None:
            print('x554: {}'.format(x554))
        elif isinstance(x554, torch.Tensor):
            print('x554: {}'.format(x554.shape))
        elif isinstance(x554, tuple):
            tuple_shapes = '('
            for item in x554:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x554: {}'.format(tuple_shapes))
        else:
            print('x554: {}'.format(x554))
        x555=self.layernorm48(x554)
        if x555 is None:
            print('x555: {}'.format(x555))
        elif isinstance(x555, torch.Tensor):
            print('x555: {}'.format(x555.shape))
        elif isinstance(x555, tuple):
            tuple_shapes = '('
            for item in x555:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x555: {}'.format(tuple_shapes))
        else:
            print('x555: {}'.format(x555))
        x558=operator.getitem(self.relative_position_bias_table22, self.relative_position_index22)
        if x558 is None:
            print('x558: {}'.format(x558))
        elif isinstance(x558, torch.Tensor):
            print('x558: {}'.format(x558.shape))
        elif isinstance(x558, tuple):
            tuple_shapes = '('
            for item in x558:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x558: {}'.format(tuple_shapes))
        else:
            print('x558: {}'.format(x558))
        x559=x558.view(49, 49, -1)
        if x559 is None:
            print('x559: {}'.format(x559))
        elif isinstance(x559, torch.Tensor):
            print('x559: {}'.format(x559.shape))
        elif isinstance(x559, tuple):
            tuple_shapes = '('
            for item in x559:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x559: {}'.format(tuple_shapes))
        else:
            print('x559: {}'.format(x559))
        x560=x559.permute(2, 0, 1)
        if x560 is None:
            print('x560: {}'.format(x560))
        elif isinstance(x560, torch.Tensor):
            print('x560: {}'.format(x560.shape))
        elif isinstance(x560, tuple):
            tuple_shapes = '('
            for item in x560:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x560: {}'.format(tuple_shapes))
        else:
            print('x560: {}'.format(x560))
        x561=x560.contiguous()
        if x561 is None:
            print('x561: {}'.format(x561))
        elif isinstance(x561, torch.Tensor):
            print('x561: {}'.format(x561.shape))
        elif isinstance(x561, tuple):
            tuple_shapes = '('
            for item in x561:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x561: {}'.format(tuple_shapes))
        else:
            print('x561: {}'.format(x561))
        x562=x561.unsqueeze(0)
        if x562 is None:
            print('x562: {}'.format(x562))
        elif isinstance(x562, torch.Tensor):
            print('x562: {}'.format(x562.shape))
        elif isinstance(x562, tuple):
            tuple_shapes = '('
            for item in x562:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x562: {}'.format(tuple_shapes))
        else:
            print('x562: {}'.format(x562))
        x567=torchvision.models.swin_transformer.shifted_window_attention(x555, self.weight44, self.weight45, x562, [7, 7], 32,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias44, proj_bias=self.bias45)
        if x567 is None:
            print('x567: {}'.format(x567))
        elif isinstance(x567, torch.Tensor):
            print('x567: {}'.format(x567.shape))
        elif isinstance(x567, tuple):
            tuple_shapes = '('
            for item in x567:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x567: {}'.format(tuple_shapes))
        else:
            print('x567: {}'.format(x567))
        x568=stochastic_depth(x567, 0.4782608695652174, 'row', False)
        if x568 is None:
            print('x568: {}'.format(x568))
        elif isinstance(x568, torch.Tensor):
            print('x568: {}'.format(x568.shape))
        elif isinstance(x568, tuple):
            tuple_shapes = '('
            for item in x568:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x568: {}'.format(tuple_shapes))
        else:
            print('x568: {}'.format(x568))
        x569=operator.add(x554, x568)
        if x569 is None:
            print('x569: {}'.format(x569))
        elif isinstance(x569, torch.Tensor):
            print('x569: {}'.format(x569.shape))
        elif isinstance(x569, tuple):
            tuple_shapes = '('
            for item in x569:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x569: {}'.format(tuple_shapes))
        else:
            print('x569: {}'.format(x569))
        x570=self.layernorm49(x569)
        if x570 is None:
            print('x570: {}'.format(x570))
        elif isinstance(x570, torch.Tensor):
            print('x570: {}'.format(x570.shape))
        elif isinstance(x570, tuple):
            tuple_shapes = '('
            for item in x570:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x570: {}'.format(tuple_shapes))
        else:
            print('x570: {}'.format(x570))
        x571=self.linear47(x570)
        if x571 is None:
            print('x571: {}'.format(x571))
        elif isinstance(x571, torch.Tensor):
            print('x571: {}'.format(x571.shape))
        elif isinstance(x571, tuple):
            tuple_shapes = '('
            for item in x571:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x571: {}'.format(tuple_shapes))
        else:
            print('x571: {}'.format(x571))
        x572=self.gelu22(x571)
        if x572 is None:
            print('x572: {}'.format(x572))
        elif isinstance(x572, torch.Tensor):
            print('x572: {}'.format(x572.shape))
        elif isinstance(x572, tuple):
            tuple_shapes = '('
            for item in x572:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x572: {}'.format(tuple_shapes))
        else:
            print('x572: {}'.format(x572))
        x573=self.dropout44(x572)
        if x573 is None:
            print('x573: {}'.format(x573))
        elif isinstance(x573, torch.Tensor):
            print('x573: {}'.format(x573.shape))
        elif isinstance(x573, tuple):
            tuple_shapes = '('
            for item in x573:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x573: {}'.format(tuple_shapes))
        else:
            print('x573: {}'.format(x573))
        x574=self.linear48(x573)
        if x574 is None:
            print('x574: {}'.format(x574))
        elif isinstance(x574, torch.Tensor):
            print('x574: {}'.format(x574.shape))
        elif isinstance(x574, tuple):
            tuple_shapes = '('
            for item in x574:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x574: {}'.format(tuple_shapes))
        else:
            print('x574: {}'.format(x574))
        x575=self.dropout45(x574)
        if x575 is None:
            print('x575: {}'.format(x575))
        elif isinstance(x575, torch.Tensor):
            print('x575: {}'.format(x575.shape))
        elif isinstance(x575, tuple):
            tuple_shapes = '('
            for item in x575:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x575: {}'.format(tuple_shapes))
        else:
            print('x575: {}'.format(x575))
        x576=stochastic_depth(x575, 0.4782608695652174, 'row', False)
        if x576 is None:
            print('x576: {}'.format(x576))
        elif isinstance(x576, torch.Tensor):
            print('x576: {}'.format(x576.shape))
        elif isinstance(x576, tuple):
            tuple_shapes = '('
            for item in x576:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x576: {}'.format(tuple_shapes))
        else:
            print('x576: {}'.format(x576))
        x577=operator.add(x569, x576)
        if x577 is None:
            print('x577: {}'.format(x577))
        elif isinstance(x577, torch.Tensor):
            print('x577: {}'.format(x577.shape))
        elif isinstance(x577, tuple):
            tuple_shapes = '('
            for item in x577:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x577: {}'.format(tuple_shapes))
        else:
            print('x577: {}'.format(x577))
        x578=self.layernorm50(x577)
        if x578 is None:
            print('x578: {}'.format(x578))
        elif isinstance(x578, torch.Tensor):
            print('x578: {}'.format(x578.shape))
        elif isinstance(x578, tuple):
            tuple_shapes = '('
            for item in x578:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x578: {}'.format(tuple_shapes))
        else:
            print('x578: {}'.format(x578))
        x581=operator.getitem(self.relative_position_bias_table23, self.relative_position_index23)
        if x581 is None:
            print('x581: {}'.format(x581))
        elif isinstance(x581, torch.Tensor):
            print('x581: {}'.format(x581.shape))
        elif isinstance(x581, tuple):
            tuple_shapes = '('
            for item in x581:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x581: {}'.format(tuple_shapes))
        else:
            print('x581: {}'.format(x581))
        x582=x581.view(49, 49, -1)
        if x582 is None:
            print('x582: {}'.format(x582))
        elif isinstance(x582, torch.Tensor):
            print('x582: {}'.format(x582.shape))
        elif isinstance(x582, tuple):
            tuple_shapes = '('
            for item in x582:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x582: {}'.format(tuple_shapes))
        else:
            print('x582: {}'.format(x582))
        x583=x582.permute(2, 0, 1)
        if x583 is None:
            print('x583: {}'.format(x583))
        elif isinstance(x583, torch.Tensor):
            print('x583: {}'.format(x583.shape))
        elif isinstance(x583, tuple):
            tuple_shapes = '('
            for item in x583:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x583: {}'.format(tuple_shapes))
        else:
            print('x583: {}'.format(x583))
        x584=x583.contiguous()
        if x584 is None:
            print('x584: {}'.format(x584))
        elif isinstance(x584, torch.Tensor):
            print('x584: {}'.format(x584.shape))
        elif isinstance(x584, tuple):
            tuple_shapes = '('
            for item in x584:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x584: {}'.format(tuple_shapes))
        else:
            print('x584: {}'.format(x584))
        x585=x584.unsqueeze(0)
        if x585 is None:
            print('x585: {}'.format(x585))
        elif isinstance(x585, torch.Tensor):
            print('x585: {}'.format(x585.shape))
        elif isinstance(x585, tuple):
            tuple_shapes = '('
            for item in x585:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x585: {}'.format(tuple_shapes))
        else:
            print('x585: {}'.format(x585))
        x590=torchvision.models.swin_transformer.shifted_window_attention(x578, self.weight46, self.weight47, x585, [7, 7], 32,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias46, proj_bias=self.bias47)
        if x590 is None:
            print('x590: {}'.format(x590))
        elif isinstance(x590, torch.Tensor):
            print('x590: {}'.format(x590.shape))
        elif isinstance(x590, tuple):
            tuple_shapes = '('
            for item in x590:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x590: {}'.format(tuple_shapes))
        else:
            print('x590: {}'.format(x590))
        x591=stochastic_depth(x590, 0.5, 'row', False)
        if x591 is None:
            print('x591: {}'.format(x591))
        elif isinstance(x591, torch.Tensor):
            print('x591: {}'.format(x591.shape))
        elif isinstance(x591, tuple):
            tuple_shapes = '('
            for item in x591:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x591: {}'.format(tuple_shapes))
        else:
            print('x591: {}'.format(x591))
        x592=operator.add(x577, x591)
        if x592 is None:
            print('x592: {}'.format(x592))
        elif isinstance(x592, torch.Tensor):
            print('x592: {}'.format(x592.shape))
        elif isinstance(x592, tuple):
            tuple_shapes = '('
            for item in x592:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x592: {}'.format(tuple_shapes))
        else:
            print('x592: {}'.format(x592))
        x593=self.layernorm51(x592)
        if x593 is None:
            print('x593: {}'.format(x593))
        elif isinstance(x593, torch.Tensor):
            print('x593: {}'.format(x593.shape))
        elif isinstance(x593, tuple):
            tuple_shapes = '('
            for item in x593:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x593: {}'.format(tuple_shapes))
        else:
            print('x593: {}'.format(x593))
        x594=self.linear49(x593)
        if x594 is None:
            print('x594: {}'.format(x594))
        elif isinstance(x594, torch.Tensor):
            print('x594: {}'.format(x594.shape))
        elif isinstance(x594, tuple):
            tuple_shapes = '('
            for item in x594:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x594: {}'.format(tuple_shapes))
        else:
            print('x594: {}'.format(x594))
        x595=self.gelu23(x594)
        if x595 is None:
            print('x595: {}'.format(x595))
        elif isinstance(x595, torch.Tensor):
            print('x595: {}'.format(x595.shape))
        elif isinstance(x595, tuple):
            tuple_shapes = '('
            for item in x595:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x595: {}'.format(tuple_shapes))
        else:
            print('x595: {}'.format(x595))
        x596=self.dropout46(x595)
        if x596 is None:
            print('x596: {}'.format(x596))
        elif isinstance(x596, torch.Tensor):
            print('x596: {}'.format(x596.shape))
        elif isinstance(x596, tuple):
            tuple_shapes = '('
            for item in x596:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x596: {}'.format(tuple_shapes))
        else:
            print('x596: {}'.format(x596))
        x597=self.linear50(x596)
        if x597 is None:
            print('x597: {}'.format(x597))
        elif isinstance(x597, torch.Tensor):
            print('x597: {}'.format(x597.shape))
        elif isinstance(x597, tuple):
            tuple_shapes = '('
            for item in x597:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x597: {}'.format(tuple_shapes))
        else:
            print('x597: {}'.format(x597))
        x598=self.dropout47(x597)
        if x598 is None:
            print('x598: {}'.format(x598))
        elif isinstance(x598, torch.Tensor):
            print('x598: {}'.format(x598.shape))
        elif isinstance(x598, tuple):
            tuple_shapes = '('
            for item in x598:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x598: {}'.format(tuple_shapes))
        else:
            print('x598: {}'.format(x598))
        x599=stochastic_depth(x598, 0.5, 'row', False)
        if x599 is None:
            print('x599: {}'.format(x599))
        elif isinstance(x599, torch.Tensor):
            print('x599: {}'.format(x599.shape))
        elif isinstance(x599, tuple):
            tuple_shapes = '('
            for item in x599:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x599: {}'.format(tuple_shapes))
        else:
            print('x599: {}'.format(x599))
        x600=operator.add(x592, x599)
        if x600 is None:
            print('x600: {}'.format(x600))
        elif isinstance(x600, torch.Tensor):
            print('x600: {}'.format(x600.shape))
        elif isinstance(x600, tuple):
            tuple_shapes = '('
            for item in x600:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x600: {}'.format(tuple_shapes))
        else:
            print('x600: {}'.format(x600))
        x601=self.layernorm52(x600)
        if x601 is None:
            print('x601: {}'.format(x601))
        elif isinstance(x601, torch.Tensor):
            print('x601: {}'.format(x601.shape))
        elif isinstance(x601, tuple):
            tuple_shapes = '('
            for item in x601:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x601: {}'.format(tuple_shapes))
        else:
            print('x601: {}'.format(x601))
        x602=x601.permute(0, 3, 1, 2)
        if x602 is None:
            print('x602: {}'.format(x602))
        elif isinstance(x602, torch.Tensor):
            print('x602: {}'.format(x602.shape))
        elif isinstance(x602, tuple):
            tuple_shapes = '('
            for item in x602:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x602: {}'.format(tuple_shapes))
        else:
            print('x602: {}'.format(x602))
        x603=self.adaptiveavgpool2d0(x602)
        if x603 is None:
            print('x603: {}'.format(x603))
        elif isinstance(x603, torch.Tensor):
            print('x603: {}'.format(x603.shape))
        elif isinstance(x603, tuple):
            tuple_shapes = '('
            for item in x603:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x603: {}'.format(tuple_shapes))
        else:
            print('x603: {}'.format(x603))
        x604=torch.flatten(x603, 1)
        if x604 is None:
            print('x604: {}'.format(x604))
        elif isinstance(x604, torch.Tensor):
            print('x604: {}'.format(x604.shape))
        elif isinstance(x604, tuple):
            tuple_shapes = '('
            for item in x604:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x604: {}'.format(tuple_shapes))
        else:
            print('x604: {}'.format(x604))
        x605=self.linear51(x604)
        if x605 is None:
            print('x605: {}'.format(x605))
        elif isinstance(x605, torch.Tensor):
            print('x605: {}'.format(x605.shape))
        elif isinstance(x605, tuple):
            tuple_shapes = '('
            for item in x605:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x605: {}'.format(tuple_shapes))
        else:
            print('x605: {}'.format(x605))

m = M().eval()
CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x = torch.rand(1, 3, 224, 224)
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
