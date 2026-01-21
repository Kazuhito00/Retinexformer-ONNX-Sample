"""
Google DriveからPyTorch事前学習済みモデル（.pth）をダウンロードするスクリプト
"""

import argparse
import os
from pathlib import Path
import sys

try:
    import gdown
except ImportError:
    print("Error: gdown is not installed.")
    print("Please install it with: pip install gdown")
    sys.exit(1)


# Google Drive上の事前学習済みモデルフォルダ
# https://drive.google.com/drive/folders/1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV?usp=drive_link
PRETRAINED_FOLDER_ID = "1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV"

# 各モデルの個別ファイルID（フォルダからダウンロードした後に取得可能）
MODEL_FILES = [
    "LOL_v1.pth",
    "LOL_v2_real.pth",
    "LOL_v2_synthetic.pth",
    "SID.pth",
    "SMID.pth",
    "SDSD_indoor.pth",
    "SDSD_outdoor.pth",
    "FiveK.pth",
    "NTIRE.pth",
]


def download_folder(folder_id: str, output_dir: str):
    """
    Google Driveフォルダ全体をダウンロード

    Args:
        folder_id: GoogleドライブフォルダID
        output_dir: 保存先ディレクトリ
    """
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    print(f"Downloading pretrained models from Google Drive...")
    print(f"Source: {url}")
    print(f"Destination: {output_dir}")
    print("-" * 60)

    # 出力ディレクトリを作成
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        gdown.download_folder(
            url,
            output=output_dir,
            quiet=False,
            use_cookies=False,
            remaining_ok=True
        )
        print("-" * 60)
        print("✓ Download completed!")

        # ダウンロードされたファイルをリスト表示
        list_downloaded_models(output_dir)

    except Exception as e:
        print(f"✗ Failed to download: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify the folder is publicly accessible")
        print("3. Try updating gdown: pip install --upgrade gdown")
        sys.exit(1)


def list_downloaded_models(output_dir: str):
    """
    ダウンロードされたモデルファイルをリスト表示

    Args:
        output_dir: モデルが保存されているディレクトリ
    """
    print("\nDownloaded models:")
    pth_files = list(Path(output_dir).glob("**/*.pth"))

    if not pth_files:
        print("  No .pth files found")
        return

    total_size = 0
    for i, pth_file in enumerate(sorted(pth_files), 1):
        size_mb = pth_file.stat().st_size / (1024 * 1024)
        total_size += size_mb
        rel_path = pth_file.relative_to(output_dir)
        print(f"  {i}. {rel_path} ({size_mb:.1f} MB)")

    print(f"\nTotal: {len(pth_files)} models, {total_size:.1f} MB")


def check_existing_models(output_dir: str):
    """
    既存のモデルをチェック

    Args:
        output_dir: チェックするディレクトリ

    Returns:
        bool: .pthファイルが存在するか
    """
    if not os.path.exists(output_dir):
        return False

    pth_files = list(Path(output_dir).glob("**/*.pth"))
    return len(pth_files) > 0


def main():
    parser = argparse.ArgumentParser(
        description="Download PyTorch pretrained models (.pth) from Google Drive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all pretrained models
  python download_pretrained_models.py

  # Download to a specific directory
  python download_pretrained_models.py --output weights

  # List existing models without downloading
  python download_pretrained_models.py --list

  # Force re-download even if files exist
  python download_pretrained_models.py --force
"""
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pretrained_weights",
        help="Output directory (default: pretrained_weights)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List existing models without downloading"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist"
    )

    args = parser.parse_args()

    # リストモード
    if args.list:
        if check_existing_models(args.output):
            list_downloaded_models(args.output)
        else:
            print(f"No models found in '{args.output}'")
        return

    # 既存ファイルのチェック
    if check_existing_models(args.output) and not args.force:
        print(f"Models already exist in '{args.output}'")
        list_downloaded_models(args.output)
        print("\nOptions:")
        print("  - Use --force to re-download")
        print("  - Use --list to view existing models")
        response = input("\nContinue downloading anyway? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # ダウンロード実行
    download_folder(PRETRAINED_FOLDER_ID, args.output)


if __name__ == "__main__":
    main()
