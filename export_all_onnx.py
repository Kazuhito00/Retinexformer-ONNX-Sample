"""
すべてのRetinexFormerモデルをONNX形式にエクスポート
"""

import os
import torch
import yaml
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from basicsr.models.archs.RetinexFormer_arch import RetinexFormer


# モデルとオプションファイルのマッピング
MODEL_CONFIGS = [
    {
        "name": "LOL_v1",
        "opt": "Options/RetinexFormer_LOL_v1.yml",
        "weights": "pretrained_weights/LOL_v1.pth",
    },
    {
        "name": "LOL_v2_real",
        "opt": "Options/RetinexFormer_LOL_v2_real.yml",
        "weights": "pretrained_weights/LOL_v2_real.pth",
    },
    {
        "name": "LOL_v2_synthetic",
        "opt": "Options/RetinexFormer_LOL_v2_synthetic.yml",
        "weights": "pretrained_weights/LOL_v2_synthetic.pth",
    },
    {
        "name": "SID",
        "opt": "Options/RetinexFormer_SID.yml",
        "weights": "pretrained_weights/SID.pth",
    },
    {
        "name": "SMID",
        "opt": "Options/RetinexFormer_SMID.yml",
        "weights": "pretrained_weights/SMID.pth",
    },
    {
        "name": "SDSD_indoor",
        "opt": "Options/RetinexFormer_SDSD_indoor.yml",
        "weights": "pretrained_weights/SDSD_indoor.pth",
    },
    {
        "name": "SDSD_outdoor",
        "opt": "Options/RetinexFormer_SDSD_outdoor.yml",
        "weights": "pretrained_weights/SDSD_outdoor.pth",
    },
    {
        "name": "FiveK",
        "opt": "Options/RetinexFormer_FiveK.yml",
        "weights": "pretrained_weights/FiveK.pth",
    },
    {
        "name": "NTIRE",
        "opt": "Options/RetinexFormer_NTIRE.yml",
        "weights": "pretrained_weights/NTIRE.pth",
    },
]


def export_model(opt_path: str, weights_path: str, output_path: str):
    """
    単一モデルをONNXにエクスポート
    """
    # YAMLファイルからモデル設定を読み込む（UTF-8エンコーディング）
    with open(opt_path, 'r', encoding='utf-8') as f:
        opt = yaml.safe_load(f)

    network_g = opt.get('network_g', {})

    # モデルを作成
    model = RetinexFormer(
        in_channels=network_g.get('in_channels', 3),
        out_channels=network_g.get('out_channels', 3),
        n_feat=network_g.get('n_feat', 40),
        stage=network_g.get('stage', 1),
        num_blocks=network_g.get('num_blocks', [1, 2, 2])
    )

    # 重みをロード
    checkpoint = torch.load(weights_path, map_location='cpu')
    state_dict = checkpoint.get('params', checkpoint)

    # 'module.' プレフィックスを削除
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    # ダミー入力（4の倍数）
    dummy_input = torch.randn(1, 3, 480, 640)

    # ONNXにエクスポート
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'output': {0: 'batch', 2: 'height', 3: 'width'}
        },
        dynamo=False
    )

    return True


def main():
    # 出力ディレクトリ
    output_dir = Path("onnx_models")
    output_dir.mkdir(exist_ok=True)

    print(f"Exporting {len(MODEL_CONFIGS)} models to ONNX...")
    print(f"Output directory: {output_dir.absolute()}")
    print("-" * 60)

    success_count = 0
    failed_models = []

    for config in MODEL_CONFIGS:
        name = config["name"]
        opt_path = config["opt"]
        weights_path = config["weights"]
        output_path = output_dir / f"retinexformer_{name.lower()}.onnx"

        # ファイル存在確認
        if not os.path.exists(opt_path):
            print(f"[SKIP] {name}: Option file not found: {opt_path}")
            failed_models.append(name)
            continue

        if not os.path.exists(weights_path):
            print(f"[SKIP] {name}: Weights file not found: {weights_path}")
            failed_models.append(name)
            continue

        try:
            print(f"[EXPORTING] {name}...", end=" ", flush=True)
            export_model(opt_path, weights_path, str(output_path))
            print(f"OK -> {output_path}")
            success_count += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed_models.append(name)

    print("-" * 60)
    print(f"Export completed: {success_count}/{len(MODEL_CONFIGS)} models")

    if failed_models:
        print(f"Failed models: {', '.join(failed_models)}")


if __name__ == "__main__":
    main()
