#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, tempfile, shutil, subprocess, gc, atexit, asyncio, copy
from typing import List
import numpy as np
import cv2
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from tqdm import tqdm

# ---- Wav2Lip リポのモジュール（PYTHONPATH を Wav2Lip ルートに通しておく）----
import face_detection
from models import Wav2Lip
import audio as audio_lib  # 同梱の audio.py

# =========================================================
# 設定
# =========================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Wav2Lip本体
FACE_DEVICE = os.environ.get("W2L_FACE_DEVICE", "cpu")   # 顔検出（既定: CPU）
CKPT_PATH = os.environ.get("W2L_CKPT", "checkpoints/wav2lip_gan.pth")

IMG_SIZE = 96
MEL_STEP_SIZE = 16

DEFAULTS = dict(
    pads=[0,10,0,0],
    face_det_batch_size=16,
    wav2lip_batch_size=64,  # メモリ抑制のため既定値を少し小さめに
    resize_factor=1,
    crop=[0,-1,0,-1],
    box=[-1,-1,-1,-1],
    rotate=False,
    nosmooth=False,
    fps=25.0,
    static=None,  # Noneなら自動判定
)

# 作業用ベースディレクトリ（カレント/working_dir に固定）
BASE_WORK_DIR = os.path.join(os.getcwd(), "working_dir")
os.makedirs(BASE_WORK_DIR, exist_ok=True)

# =========================================================
# アプリ & グローバル
# =========================================================
app = FastAPI(title="Wav2Lip Service", root_path="/wav2lip_api")
MODEL = None
_infer_lock = asyncio.Semaphore(1)  # 同時推論を直列化

# =========================================================
# 初期化と後始末
# =========================================================
def _load_checkpoint(checkpoint_path: str):
    if DEVICE == 'cuda':
        return torch.load(checkpoint_path)
    else:
        return torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

def load_model_once(path: str):
    model = Wav2Lip()
    ckpt = _load_checkpoint(path)
    s = ckpt["state_dict"]
    new_s = {k.replace('module.', ''): v for k, v in s.items()}
    model.load_state_dict(new_s)
    return model.to(DEVICE).eval()

@app.on_event("startup")
def _startup():
    global MODEL
    MODEL = load_model_once(CKPT_PATH)
    print(f"[startup] Wav2Lip on {DEVICE}, FaceDetector on {FACE_DEVICE}", flush=True)
    print(f"[startup] Loaded checkpoint: {CKPT_PATH}", flush=True)

@atexit.register
def _cleanup():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

# =========================================================
# ユーティリティ
# =========================================================
def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        window = boxes[i: i+T] if i+T <= len(boxes) else boxes[len(boxes)-T:]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def face_detect(images, pads: List[int], batch_size: int, nosmooth: bool):
    """毎回ローカルで FaceAlignment を生成して破棄 → 状態・履歴汚染を防ぐ"""
    detector = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D,
        flip_input=False,
        device=FACE_DEVICE
    )

    local_bs = batch_size
    preds = []
    try:
        while True:
            try:
                preds = []
                for i in range(0, len(images), local_bs):
                    preds.extend(detector.get_detections_for_batch(np.array(images[i:i+local_bs])))
            except RuntimeError:
                if local_bs == 1:
                    raise RuntimeError('Image too big for face-detector. Try resize_factor.')
                local_bs //= 2
                print(f"[warn] face-det OOM -> retry batch={local_bs}", flush=True)
                continue
            break

        results = []
        pady1, pady2, padx1, padx2 = pads
        for rect, image in zip(preds, images):
            if rect is None:
                raise ValueError('Face not detected in some frames.')
            y1 = max(0, rect[1] - pady1); y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1); x2 = min(image.shape[1], rect[2] + padx2)
            results.append([x1, y1, x2, y2])
        boxes = np.array(results)
        if not nosmooth:
            boxes = get_smoothened_boxes(boxes, T=5)
        out = [[im[y1:y2, x1:x2], (y1, y2, x1, x2)] for im, (x1, y1, x2, y2) in zip(images, boxes)]
        return out
    finally:
        del detector
        gc.collect()

def datagen(frames, mels, p):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    static = p['static']; box = p['box']; img_size = p['img_size']
    pads = p['pads']; face_det_bs = p['face_det_batch_size']
    nosmooth = p['nosmooth']; batch_size = p['wav2lip_batch_size']

    if box[0] == -1:
        face_det_results = face_detect(frames if not static else [frames[0]], pads, face_det_bs, nosmooth)
    else:
        y1, y2, x1, x2 = box
        face_det_results = [[f[y1:y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

    for i, m in enumerate(mels):
        idx = 0 if static else i % len(frames)
        fsave = frames[idx].copy()
        face, coords = face_det_results[idx].copy()
        face = cv2.resize(face, (img_size, img_size))
        img_batch.append(face); mel_batch.append(m); frame_batch.append(fsave); coords_batch.append(coords)
        if len(img_batch) >= batch_size:
            img_batch_np, mel_batch_np = np.asarray(img_batch), np.asarray(mel_batch)
            img_masked = img_batch_np.copy(); img_masked[:, img_size//2:] = 0
            img_batch_np = np.concatenate((img_masked, img_batch_np), axis=3) / 255.
            mel_batch_np = np.reshape(mel_batch_np, [len(mel_batch_np), mel_batch_np.shape[1], mel_batch_np.shape[2], 1])
            yield img_batch_np, mel_batch_np, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    if len(img_batch) > 0:
        img_batch_np, mel_batch_np = np.asarray(img_batch), np.asarray(mel_batch)
        img_masked = img_batch_np.copy(); img_masked[:, img_size//2:] = 0
        img_batch_np = np.concatenate((img_masked, img_batch_np), axis=3) / 255.
        mel_batch_np = np.reshape(mel_batch_np, [len(mel_batch_np), mel_batch_np.shape[1], mel_batch_np.shape[2], 1])
        yield img_batch_np, mel_batch_np, frame_batch, coords_batch

def _detect_static(face_path: str) -> bool:
    ext = os.path.splitext(face_path.lower())[1].lstrip('.')
    if ext in ('jpg','jpeg','png'):
        return True
    img_try = cv2.imread(face_path)
    return (img_try is not None)

def run_wav2lip(face_path: str, audio_path: str, out_mp4: str, user_params: dict) -> str:
    # ---- dictは deep copy してから上書き（リスト汚染防止）----
    p = copy.deepcopy(DEFAULTS)
    p.update(user_params or {})
    p['img_size'] = IMG_SIZE

    if p['static'] is None:
        p['static'] = _detect_static(face_path)

    if not os.path.isfile(face_path):
        raise ValueError('face path invalid')

    # ---- 入力フレーム取得 ----
    if p['static']:
        full_frames = [cv2.imread(face_path)]
        fps = float(p['fps'])
    else:
        vs = cv2.VideoCapture(face_path); fps = vs.get(cv2.CAP_PROP_FPS); full_frames=[]
        try:
            while True:
                ok, frame = vs.read()
                if not ok: break
                if p['resize_factor'] > 1:
                    frame = cv2.resize(frame, (frame.shape[1]//p['resize_factor'], frame.shape[0]//p['resize_factor']))
                if p['rotate']:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                y1, y2, x1, x2 = p['crop']
                if x2 == -1: x2 = frame.shape[1]
                if y2 == -1: y2 = frame.shape[0]
                frame = frame[y1:y2, x1:x2]
                full_frames.append(frame)
        finally:
            vs.release()

    # ---- 音声: 16k/mono に整形 ----
    workdir = tempfile.mkdtemp(prefix='w2l_', dir=BASE_WORK_DIR)
    tmp_wav = os.path.join(workdir, 'audio.wav')
    subprocess.check_call(f'ffmpeg -y -i "{audio_path}" -ar 16000 -ac 1 -f wav "{tmp_wav}"', shell=True)

    wav = audio_lib.load_wav(tmp_wav, 16000)
    mel = audio_lib.melspectrogram(wav)
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains NaN. Try adding small noise to wav.')

    # ---- mel をフレーム数に合わせて分割 ----
    mel_chunks = []
    mel_idx_multiplier = 80. / (p['fps'] if p['static'] else (fps or p['fps']))
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + MEL_STEP_SIZE > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - MEL_STEP_SIZE:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + MEL_STEP_SIZE])
        i += 1

    full_frames = full_frames[:len(mel_chunks)]

    # ---- 合成 ----
    frame_h, frame_w = full_frames[0].shape[:-1]
    temp_avi = os.path.join(workdir, 'result.avi')
    outv = cv2.VideoWriter(
        temp_avi, cv2.VideoWriter_fourcc(*'DIVX'),
        float(p['fps'] if p['static'] else (fps or p['fps'])),
        (frame_w, frame_h)
    )

    gen = datagen(full_frames.copy(), mel_chunks, p)

    # --- 同期的・安全版 推論ループ（非同期コピー禁止） ---
    total_steps = int(np.ceil(len(mel_chunks)/p['wav2lip_batch_size']))
    for img_batch, mel_batch, frames, coords in tqdm(gen, total=total_steps, desc='infer', leave=False):
        # CPU -> GPU は同期（non_blocking=False）
        img_np = np.transpose(img_batch, (0,3,1,2)).astype(np.float32)
        mel_np = np.transpose(mel_batch, (0,3,1,2)).astype(np.float32)
        img_t = torch.from_numpy(img_np).to(DEVICE)
        mel_t = torch.from_numpy(mel_np).to(DEVICE)

        with torch.inference_mode():
            pred = MODEL(mel_t, img_t)

        # ここでGPU計算を完了させる
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # CPU に同期コピー（non_blocking=False）
        pred_cpu = pred.detach().to('cpu')
        pred_np = pred_cpu.numpy().transpose(0,2,3,1) * 255.0

        # 直ちにGPUテンソルを解放
        del pred, img_t, mel_t, pred_cpu

        # 合成
        for pimg, f, c in zip(pred_np, frames, coords):
            y1, y2, x1, x2 = c
            pimg = cv2.resize(pimg.astype(np.uint8), (x2 - x1, y2 - y1))
            f[y1:y2, x1:x2] = pimg
            outv.write(f)

        # 大きい配列は都度破棄
        del pred_np, img_batch, mel_batch, frames, coords, img_np, mel_np
        gc.collect()

    outv.release()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # ---- 音声とmuxしてmp4 ----
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    subprocess.check_call(
        f'ffmpeg -y -i "{tmp_wav}" -i "{temp_avi}" -shortest -c:v libx264 -pix_fmt yuv420p '
        f'-movflags +faststart -c:a aac "{out_mp4}"',
        shell=True
    )

    # 作業ディレクトリを掃除（出力は out_mp4 側にある）
    try:
        shutil.rmtree(workdir, ignore_errors=True)
    except Exception:
        pass

    # さらに状態を残さない
    del gen, full_frames, mel_chunks
    gc.collect()

    return out_mp4

def resize_with_padding(image_path, output_path):
    # 画像読み込み
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("画像が読み込めませんでした")

    h, w = img.shape[:2]

    print("Image size: %d x %d" % (w, h))

    # 縦横比でターゲットサイズを決定
    if h > w:  # 縦長 → 720×1280
        target_w, target_h = 720, 1280
    else:       # 横長 → 1280×720 もしくは正方形
        target_w, target_h = 1280, 720

    # 縦横比を維持しながら縮小
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 黒背景キャンバス作成
    result = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # 中央に配置
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    # 保存
    cv2.imwrite(output_path, result)

# =========================================================
# エンドポイント
# =========================================================
@app.post("/lip-sync")
async def lip_sync(
    face: UploadFile = File(...),
    audio: UploadFile = File(...),
    fps: float = Form(25.0),
    pads: str = Form("0,10,0,0"),
    resize_factor: int = Form(1),
    rotate: int = Form(0),
    nosmooth: int = Form(0),
    fbs: int = Form(16),
    wbs: int = Form(64),
    crop: str = Form("0,-1,0,-1"),
    box: str = Form("-1,-1,-1,-1"),
):
    # 同時実行制御
    async with _infer_lock:
        work_in = tempfile.mkdtemp(prefix="w2l_in_", dir=BASE_WORK_DIR)
        outdir  = tempfile.mkdtemp(prefix="w2l_out_", dir=BASE_WORK_DIR)
        face_path = os.path.join(work_in, face.filename or "face_input")
        face_trans = os.path.join(work_in, "face_trans.jpg")
        audio_path = os.path.join(work_in, audio.filename or "audio_input")
        out_mp4    = os.path.join(outdir, "result.mp4")

        # 一時保存（大きなファイルでも扱えるように）
        with open(face_path, "wb") as f: shutil.copyfileobj(face.file, f)
        with open(audio_path, "wb") as f: shutil.copyfileobj(audio.file, f)

        resize_with_padding(face_path, face_trans);

        params = {
            "fps": float(fps),
            "pads": [int(x) for x in pads.split(",")],
            "resize_factor": int(resize_factor),
            "rotate": (rotate == 1),
            "nosmooth": (nosmooth == 1),
            "face_det_batch_size": int(fbs),
            "wav2lip_batch_size": int(wbs),
            "crop": [int(x) for x in crop.split(",")],
            "box":  [int(x) for x in box.split(",")],
            "static": None,
        }

        try:
            out_path = run_wav2lip(face_trans, audio_path, out_mp4, params)
            return FileResponse(out_path, media_type="video/mp4", filename="result.mp4")
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        finally:
            try:
                await face.close()
                await audio.close()
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            # 入力側の一時ディレクトリは削除（出力側は FileResponse が使うので保持）
#            try:
#                shutil.rmtree(work_in, ignore_errors=True)
#            except Exception:
#                pass
