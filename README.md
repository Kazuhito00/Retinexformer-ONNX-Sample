# Retinexformer-ONNX-Sample
Low-Light Image Enhancement モデルである [caiyuanhao1998/Retinexformer](https://github.com/caiyuanhao1998/Retinexformer) のONNX推論サンプルです。<br><br>
![Retinexformer](https://github.com/user-attachments/assets/15f1bfc7-2ece-48e2-80d1-1e9b7d8e8bff)

# Requirement
* opencv-python
* onnxruntime

# Demo
## Webカメラで実行
デフォルトでカメラID 0を使用します。
```bash
python sample_onnx_inference.py
```

## 動画ファイルで実行
```bash
python sample_onnx_inference.py --video video.mp4
```

## GPU使用
```bash
python sample_onnx_inference.py --use_gpu
```

## 出力ファイルの保存
```bash
python sample_onnx_inference.py --output output.mp4
```

# Options
* --video, -v<br>
入力動画ファイルのパス<br>
デフォルト：なし（カメラを使用）

* --camera, -c<br>
カメラデバイス番号の指定<br>
デフォルト：0

* --output, -o<br>
出力動画ファイルのパス（省略時は保存なし）<br>
デフォルト：なし

* --onnx<br>
ONNXモデルファイルのパス<br>
デフォルト：onnx_models/retinexformer_lol_v1.onnx

* --use_gpu<br>
GPU使用フラグ<br>
デフォルト：False（CPU実行）

# 操作方法
* `q` または `ESC` キー：終了

# 利用可能なモデル
onnx_modelsディレクトリに以下のモデルが含まれています：
* retinexformer_lol_v1.onnx（デフォルト）
* retinexformer_lol_v2_real.onnx
* retinexformer_lol_v2_synthetic.onnx
* retinexformer_sid.onnx
* retinexformer_smid.onnx
* retinexformer_sdsd_indoor.onnx
* retinexformer_sdsd_outdoor.onnx
* retinexformer_fivek.onnx
* retinexformer_ntire.onnx

# Reference
* [Retinexformer](https://github.com/caiyuanhao1998/Retinexformer)

# Author
高橋かずひと(https://x.com/KzhtTkhs)

# License
Retinexformer-ONNX-Sample is under MIT License.
