# 如何支持其他推理引擎 / How to Support Other Inference Engines

本文档将指导您如何为项目添加其他推理引擎的支持（以ONNXRuntime为例）。在整个项目中，唯一跟推理引擎相关的，只有predict_base.py这个文件。当更换推理引擎时，例如，将OpenVINO更换为onnxruntime时，仅需要用其它的推理引擎实现从硬盘载入模型文件，指定推理设备，并返回模型即可。

This document guides you through adding support for other inference engines (using ONNXRuntime as an example). The core project logic only interacts with the abstract class in `predict_base.py`. Switching inference engines only requires implementing three core functions: model loading, device specification, and inference execution.

## 实现步骤 / Implementation Steps

### 创建新引擎的predict_base.py / Create a New Engine's `predict_base.py`
新建`PredictBase`基类：

```python
import onnxruntime

class PredictBase(object):
    def __init__(self):
        pass

    def get_onnx_session(self, model_dir, device):

        if device = "GPU":
            providers = providers=['CUDAExecutionProvider']
        else:
            providers = providers = ['CPUExecutionProvider']

        onnx_session = onnxruntime.InferenceSession(model_dir, None,providers=providers)

        return onnx_session
```