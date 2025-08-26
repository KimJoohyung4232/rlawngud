# app.py
import io
import re
import hashlib
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import requests
from ultralytics import YOLO

# ==================== 기본 설정 ====================
st.set_page_config(page_title="YOLO 탐지기", page_icon="🧠", layout="centered")

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "best.pt"

# 🔥 GitHub Release에서 모델 받아오기(구글드라이브 X)
#   릴리스: v1.0.0, 에셋 파일명: default.pt  (네 스크린샷 기준)
MODEL_URL = "https://github.com/KimJoohyung4232/rlawngud/releases/download/v1.0.0/default.pt"
# 릴리스 페이지의 sha256 전체값을 복붙해서 넣어주세요 (스크린샷에 나온 값의 앞부분 예시)
MODEL_SHA256 = "35b9cfe36e700c3ce7f4db3f7..."  # ← 릴리스 화면의 sha256 전체 문자열로 교체!

DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 영어 → 한글 매핑
KOR_LABELS = {
    "jjajangmyeon": "짜장면",
    "jajangmyeon": "짜장면",
    "jjajang": "짜장면",
    "jajang": "짜장면",
    "blackbean_noodles": "짜장면",
    "blackbean": "짜장면",
    "ramen": "라면",
    "noodles": "면",
}
def to_kor(name: str) -> str:
    return KOR_LABELS.get(str(name).strip().lower(), name)

# ==================== 유틸 ====================
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def download_from_github_release(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)

def looks_like_html(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(512).lower()
        return b"<html" in head or b"<!doctype html" in head
    except Exception:
        return True

def likely_broken(path: Path) -> bool:
    # 1) 1MB 미만이면 거의 확실히 깨짐
    if not path.exists() or path.stat().st_size < 1_000_000:
        return True
    # 2) HTML 페이지가 저장된 경우
    if looks_like_html(path):
        return True
    # 3) SHA256이 제공되면 검증
    if MODEL_SHA256 and len(MODEL_SHA256) > 10:
        try:
            return sha256_of(path) != MODEL_SHA256.lower()
        except Exception:
            return True
    return False

# ==================== 폰트 유틸 ====================
def get_korean_font(size=18):
    font_candidates = [
        str(BASE_DIR / "fonts" / "NotoSansKR-Regular.ttf"),
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "C:/Windows/Fonts/malgun.ttf",
    ]
    for p in font_candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()

# ==================== 모델 로드 ====================
@st.cache_resource
def load_model(path: Path):
    # 모델 파일 없거나/손상 시 GitHub에서 재다운
    if likely_broken(path):
        with st.spinner("모델 다운로드 중... (GitHub Release)"):
            try:
                download_from_github_release(MODEL_URL, path)
            except Exception as e:
                st.error(f"모델 다운로드 실패: {e}")
                st.stop()
        if likely_broken(path):
            st.error("모델 파일이 손상되었습니다. 릴리스의 sha256 값과 URL을 확인해 주세요.")
            st.stop()

    # YOLO 가중치 로드
    return YOLO(str(path))

# ==================== 박스 드로잉 ====================
def draw_boxes(pil_img: Image.Image, results, names_dict, font=None):
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    font = font or get_korean_font(18)

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0].item())
            cls  = int(box.cls[0].item())
            cls_eng = names_dict.get(cls, str(cls))
            cls_name = to_kor(cls_eng)

            draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=3)
            label = f"{cls_name} {conf:.2f}"
            try:
                tw, th = draw.textbbox((0, 0), label, font=font)[2:]
            except Exception:
                tw, th = font.getsize(label)
            pad = 4
            if y1 - th - pad * 2 < 0:
                bx1, by1 = x1, y1
                bx2, by2 = x1 + tw + pad * 2, y1 + th + pad * 2
                text_xy = (x1 + pad, y1 + pad)
            else:
                bx1, by1 = x1, y1 - th - pad * 2
                bx2, by2 = x1 + tw + pad * 2, y1
                text_xy = (x1 + pad, y1 - th - pad)

            draw.rectangle([(bx1, by1), (bx2, by2)], fill=(0, 255, 0))
            draw.text(text_xy, label, font=font, fill=(0, 0, 0))
    return img

def summarize_prediction(rows):
    if not rows:
        return "아직 확신하기 어려워요. (탐지 결과 없음)"
    totals = {}
    for r in rows:
        totals[r["class_name"]] = totals.get(r["class_name"], 0.0) + float(r["conf"])
    best_name = max(totals, key=totals.get)
    best_name_kor = to_kor(best_name)
    return f'이 사진은 **"{best_name_kor}"**으로 추정됩니다.'

# ==================== UI ====================
st.title("🧠 YOLO 객체 탐지 (Streamlit)")
st.caption(f"Device: {DEVICE}")

with st.sidebar:
    st.subheader("설정")
    conf_thres = st.slider("Confidence", 0.1, 0.9, 0.25, 0.05)
    iou_thres  = st.slider("IoU", 0.1, 0.9, 0.45, 0.05)
    st.write("모델:", f"`{MODEL_PATH.name}`")
    if st.button("🔄 초기화"):
        for k in ("pred_img", "det_rows", "summary_msg", "uploaded_img"):
            st.session_state.pop(k, None)
        st.rerun()

# 업로드
st.markdown("### 1) 이미지 올리기")
up = st.file_uploader("이미지 선택 (jpg/png)", type=["jpg", "jpeg", "png"])
if up:
    pil_img = Image.open(io.BytesIO(up.read())).convert("RGB")
    st.session_state["uploaded_img"] = pil_img
    st.image(pil_img, caption="업로드 이미지", use_container_width=True)

# 버튼
st.markdown("### 2) 예측 실행")
c1, c2 = st.columns(2)
run_btn   = c1.button("🚀 예측하기", use_container_width=True)
clear_btn = c2.button("🗑 결과 지우기", use_container_width=True)

if clear_btn:
    for k in ("pred_img", "det_rows", "summary_msg"):
        st.session_state.pop(k, None)
    st.toast("결과 초기화!", icon="🧽")

# ==================== 추론 ====================
model = load_model(MODEL_PATH)

if run_btn:
    if "uploaded_img" not in st.session_state:
        st.warning("먼저 이미지를 올려줘!")
    else:
        with st.spinner("모델 추론 중..."):
            dv = "mps" if DEVICE == "mps" else (0 if DEVICE == "cuda" else "cpu")
            img_np = np.array(st.session_state["uploaded_img"])
            results = model.predict(
                source=img_np, conf=conf_thres, iou=iou_thres,
                verbose=False, device=dv
            )
            names = model.names

            out_img = draw_boxes(st.session_state["uploaded_img"], results, names)
            st.session_state["pred_img"] = out_img

            rows = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls  = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    cls_eng = names.get(cls, str(cls))
                    cls_kor = to_kor(cls_eng)
                    rows.append({
                        "class_id": cls,
                        "class_name": cls_kor,
                        "conf": round(conf, 4),
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2
                    })
            st.session_state["det_rows"]  = rows
            st.session_state["summary_msg"] = summarize_prediction(rows)

        st.success("예측 완료!")

# ==================== 결과 표시 ====================
if "pred_img" in st.session_state:
    st.markdown("### 3) 결과")
    st.image(st.session_state["pred_img"], caption="탐지 결과", use_container_width=True)

    msg = st.session_state.get("summary_msg")
    if msg:
        st.info(msg)

    if st.session_state.get("det_rows"):
        st.markdown("#### 탐지 박스 목록")
        st.json(st.session_state["det_rows"])
