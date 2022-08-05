import torchvision.models as models
import geffnet
import sys, os
from transformers import AutoModelForQuestionAnswering

vision_models = ['alexnet', 'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large', 'swin_t', 'swin_s', 'swin_b', \
                 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d', \
                 'wide_resnet50_2', 'wide_resnet101_2', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn', \
                 'squeezenet1_0', 'squeezenet1_1', 'densenet121', 'densenet169', 'densenet201', 'densenet161', 'inception_v3_google', 'googlenet', \
                 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', \
                 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', \
                 'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', \
                 'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf', 'regnet_y_8gf', 'regnet_y_16gf', 'regnet_y_32gf', \
                 'regnet_y_128gf', 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_1_6gf', 'regnet_x_3_2gf', 'regnet_x_8gf','regnet_x_16gf', 'regnet_x_32gf', \
                 'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14',
                ]
segmentation_models = ['fcn_resnet50', 'fcn_resnet101', 'deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3_mobilenet_v3_large', 'lraspp_mobilenet_v3_large']
detection_models = ['fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn_v2', 'fasterrcnn_mobilenet_v3_large_fpn', 'fasterrcnn_mobilenet_v3_large_320_fpn', \
                    'fcos_resnet50_fpn', 'maskrcnn_resnet50_fpn', 'maskrcnn_resnet50_fpn_v2', 'retinanet_resnet50_fpn', \
                    'retinanet_resnet50_fpn_v2', 'ssd300_vgg16', 'ssdlite320_mobilenet_v3_large', 'keypointrcnn_resnet50_fpn' \
                   ]
video_models = ['r3d_18', 'mc3_18', 'r2plus1d_18', 'mvit_v1_b']
optical_flow_models = ['raft_large', 'raft_small']
geffnet_models = ['fbnetc_100', 'spnasnet_100', 'efficientnet_b0_geffnet', 'efficientnet_b1_geffnet', 'efficientnet_b2_geffnet', \
                  'efficientnet_b3_geffnet', 'efficientnet_b4_geffnet', 'efficientnet_b5_geffnet', 'efficientnet_b6_geffnet', 'efficientnet_b7_geffnet' \
                 ]
qa_models = ['qa+bert-base-cased', 'qa+albert-base-v1', 'qa+roberta-base', 'qa+xlm-roberta-base', \
             'qa+google/electra-base-generator', 'qa+google/electra-base-discriminator', 'qa+distilbert-base-cased', \
             ]

def get_model(name):
    if "+" in name:
        task, model = name.split("+")
        if task == "qa":
            return AutoModelForQuestionAnswering.from_pretrained(model)
    elif name in vision_models:
        return models.__dict__[name]()
    elif name in segmentation_models:
        return models.segmentation.__dict__[name]()
    elif name in detection_models:
        return models.detection.__dict__[name]()
    elif name in video_models:
        return models.video.__dict__[name]()
    elif name in optical_flow_models:
        return models.optical_flow.__dict__[name]()
    elif name in geffnet_models:
        return geffnet.create_model(name, pretrained=True)
    else:
        print("Model " + name + " is not supported.")
        exit()

def get_inputs_dict(name, inputs):
    inputs_dict = {}
    if "+" in name:
        task, _ = name.split("+")
        if task == "qa":
            for i in range(3):
                inputs_dict[inputs[i]] = "torch.ones((1, 384), dtype=torch.long)"
            for i in range(3, len(inputs)):
                inputs_dict[inputs[i]] = "None"
    elif name in vision_models or name in geffnet_models:
        inputs_dict[inputs[0]] = "torch.rand(1, 3, 224, 224)"
    return inputs_dict
