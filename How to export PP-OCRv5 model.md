## How to Export PP-OCRv5 Models to ONNX for deployment?

### Step 1: Create & Activate the ppocrv5_ov virtual environment if not already created.
```bash
conda create -n ppocrv5_ov python=3.11
conda activate ppocrv5_ov

```

### Step 2: Install PaddlePaddle and PaddleOCR
```bash
pip install paddlepaddle
pip install paddleocr
paddleocr install_hpi_deps cpu
```

### Step 3: Install paddle2onnx
```bash
paddlex --install paddle2onnx
```

### Step 4: Download PP-OCRv5_server pre-trained model
```bash
# Download and unzip PP-OCRv5_server_det pre-trained model
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar && tar -xvf PP-OCRv5_server_det_infer.tar

# Download and upzip PP-OCRv5_server_rec pre-trained model
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_rec_infer.tar && tar -xvf PP-OCRv5_server_rec_infer.tar

# Download and upzip PP-OCRv5_server_cls pre-trained model
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar && tar -xvf PP-LCNet_x1_0_doc_ori_infer.tar

```

### Step 5: Export PP-OCRv5_server models to ONNX
```bash
# Export PP-OCRv5_server_det to ONNX
paddlex --paddle2onnx --paddle_model_dir ./PP-OCRv5_server_det_infer --onnx_model_dir ./PP-OCRv5_server_det_onnx
# Export PP-OCRv5_server_rec to ONNX
paddlex --paddle2onnx --paddle_model_dir ./PP-OCRv5_server_rec_infer --onnx_model_dir ./PP-OCRv5_server_rec_onnx
# Export PP-OCRv5_server_cls to ONNX
paddlex --paddle2onnx --paddle_model_dir ./PP-LCNet_x1_0_doc_ori_infer --onnx_model_dir ./PP-OCRv5_server_cls_onnx

```

### Step 6: Verify the exported ONNX models
```bash
# Verify PP-OCRv5_server_det & PP-OCRv5_server_rec & PP-OCRv5_server_cls ONNX model

paddleocr ocr -i ./images/handwrite_en_demo.png                        \
              --text_detection_model_name PP-OCRv5_server_det          \
              --text_detection_model_dir PP-OCRv5_server_det_onnx      \
              --text_recognition_model_name PP-OCRv5_server_rec        \
              --text_recognition_model_dir PP-OCRv5_server_rec_onnx/   \
              --enable_hpi True                                        \
              --device cpu

```









































































































