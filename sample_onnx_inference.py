import argparse
import cv2
import numpy as np
import time
import onnxruntime as ort


def create_session(onnx_path, use_gpu=True):
    # 使用可能なプロバイダーを設定
    providers = []
    if use_gpu:
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')
        if 'DmlExecutionProvider' in ort.get_available_providers():
            providers.append('DmlExecutionProvider')
    providers.append('CPUExecutionProvider')

    # セッション作成
    session = ort.InferenceSession(onnx_path, providers=providers)
    print(f"Using: {session.get_providers()}")
    return session


def preprocess_frame(frame, factor=4):
    # BGR→RGB変換、正規化
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = rgb.shape[:2]

    # factorの倍数にパディング
    H = ((h + factor - 1) // factor) * factor
    W = ((w + factor - 1) // factor) * factor

    if H != h or W != w:
        rgb = np.pad(rgb, ((0, H - h), (0, W - w), (0, 0)), mode='reflect')

    # (H,W,C) → (1,C,H,W)
    input_tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis, :, :, :]
    return input_tensor, (h, w)


def inference(session, input_tensor):
    # 推論実行
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session.run([output_name], {input_name: input_tensor})[0]


def postprocess_frame(output, original_shape):
    # パディング除去、クリップ
    h, w = original_shape
    output = np.clip(output[:, :, :h, :w], 0, 1)

    # (1,C,H,W) → (H,W,C)
    output_np = np.transpose(output[0], (1, 2, 0))

    # uint8変換、RGB→BGR
    output_uint8 = (output_np * 255).astype(np.uint8)
    return cv2.cvtColor(output_uint8, cv2.COLOR_RGB2BGR)


def draw_timing_info(image, preprocess_ms, inference_ms, postprocess_ms):
    # 処理時間を画面に描画
    total_ms = preprocess_ms + inference_ms + postprocess_ms
    lines = [
        f"Preprocess: {preprocess_ms:.1f}ms",
        f"Inference: {inference_ms:.1f}ms",
        f"Postprocess: {postprocess_ms:.1f}ms",
        f"Total: {total_ms:.1f}ms"
    ]

    y = 20
    for line in lines:
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)
        y += 20
    return image


def process_frame(session, frame):
    # 前処理
    t0 = time.perf_counter()
    input_tensor, original_shape = preprocess_frame(frame)
    t1 = time.perf_counter()

    # 推論
    output = inference(session, input_tensor)
    t2 = time.perf_counter()

    # 後処理
    enhanced = postprocess_frame(output, original_shape)
    t3 = time.perf_counter()

    # 結果を横並びで結合
    concat_frame = np.hstack([frame, enhanced])
    concat_frame = draw_timing_info(concat_frame,
                                     (t1-t0)*1000, (t2-t1)*1000, (t3-t2)*1000)
    return concat_frame


def process_video(input_path, output_path, session):
    # 動画ファイルを開く
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open: {input_path}")

    # 動画情報取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input: {w}x{h}, {fps:.1f}fps, {total} frames")

    # 出力設定
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w*2, h))

    # フレーム処理ループ
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # フレーム処理
        result = process_frame(session, frame)

        # 保存
        if out:
            out.write(result)

        # 表示
        cv2.imshow('Original | Enhanced', result)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            break

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx}/{total} frames")

    # 終了処理
    cap.release()
    if out:
        out.release()
        print(f"Saved: {output_path}")
    cv2.destroyAllWindows()


def process_camera(camera_id, output_path, session):
    # カメラを開く
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise ValueError(f"Cannot open camera: {camera_id}")

    # カメラ情報取得
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {w}x{h}")

    # 出力設定
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (w*2, h))

    print("Press 'q' or 'ESC' to quit")

    # フレーム処理ループ
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # フレーム処理
        result = process_frame(session, frame)

        # 保存
        if out:
            out.write(result)

        # 表示
        cv2.imshow('Original | Enhanced', result)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            break

    # 終了処理
    cap.release()
    if out:
        out.release()
        print(f"Saved: {output_path}")
    cv2.destroyAllWindows()


def main():
    # 引数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", help="Input video path")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera ID (default: 0)")
    parser.add_argument("--output", "-o", help="Output video path (optional)")
    parser.add_argument("--onnx", default="onnx_models/retinexformer_lol_v1.onnx", help="ONNX model path")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for inference")

    args = parser.parse_args()

    # モデル読み込み
    print(f"Loading model: {args.onnx}")
    session = create_session(args.onnx, use_gpu=args.use_gpu)

    # 処理実行
    if args.input:
        process_video(args.input, args.output, session)
    else:
        process_camera(args.camera, args.output, session)


if __name__ == "__main__":
    main()
