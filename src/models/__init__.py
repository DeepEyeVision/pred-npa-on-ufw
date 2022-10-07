import copy

import numpy as np
import torch
import torch.nn as nn
import torch.quantization.quantize_fx as quantize_fx
from efficientnet_pytorch import EfficientNet

try:
    from openvino.inference_engine import IECore
except:
    print("IECore import failed. You may run this in Docker")
    pass

from . import PraNet_Res2Net


def get_model(arch, resume=None):
    if resume != None and resume.endswith(".xml"):
        model = VinoIE(resume)
    else:
        model = globals()[arch].get_model(resume=resume)

    return model


def set_quantize(model):
    """quantization aware training for static quantization"""
    qconfig_dict = {"": torch.quantization.get_default_qat_qconfig("qnnpack")}
    model_pre = quantize_fx.prepare_qat_fx(model.train(), qconfig_dict)
    return model_pre


def quantize(model):
    model_quantized = quantize_fx.convert_fx(copy.deepcopy(model))
    return model_quantized


class VinoIE(nn.Module):
    def __init__(self, model_path, num_threads=1):
        super().__init__()

        ie = IECore()
        ie.register_plugins(model_path)
        net = ie.read_network(
            model=model_path, weights=model_path.replace(".xml", ".bin")
        )
        self.exec_net = ie.load_network(
            network=net, device_name="CPU", num_requests=num_threads
        )
        print("self.exec_net = \n", self.exec_net.input_info)
        print(self.exec_net.outputs)

    def forward(self, x):
        output = self.exec_net.infer(inputs={"input": x.numpy()})["output"]
        return torch.from_numpy(output.astype(np.float32))
