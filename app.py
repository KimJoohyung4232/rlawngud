# app.py
import io
import re
import os
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import requests
import gdown
from ultralytics import YOLO

# ==================== 기본 설정 ====================
st.set_page_config(page_title="YOLO 탐지기", page_icon="🧠", layout="centered")

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "best.pt"   # 캐시 저장 위치

# ✅ 구글 드라이브 공유 링크의 파일 ID (너가 방금 준 새 링크)
# https://drive.google.com/file/d/13Gpp2rOV24l8-_u3QtNlASZIlTRR3v7S/view?usp=share_link
GDRIVE_FILE_ID = "13Gpp2rOV24l8-_u3QtNlASZIlTRR3v7S"

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
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

# ==================== 안전 다운로드/검증 유틸 ====================
def looks_like_html(file_path: Path, check_bytes: int = 2048) -> bool:
    """파일 앞부분이 HTML처럼 보이면 True (드라이브 경고/미리보기 페이지가 저장된 경우 방지)"""
    try:
        with open(file_path, "rb") as f:
            head = f.read(check_bytes)
        # 간단한 휴리스틱
        head_l = head.lower()
        return (
            head_l.startswith(b"<!doctype html")
            or b"<html" in head_l[:512]
            or b"google" in head_l and b"drive" in head_l and b"<html" in head_l
        )
    except Exception:
        return False

def likely_broken(file_path: Path) -> bool:
    """사이즈가 비정상적으로 작거나/HTML이면 손상으로 판단"""
    if not file_path.exists():
        return True
    # pt가 1MB 미만이면 99% 이상 손상/HTML
    if file_path.stat().st_size < 1_000_000:
        return True
    if looks_like_html(file_path):
        return True
    return False

def download_model_from_gdrive(file_id: str, dst: Path):
    """gdown을 사용해서 Drive에서 확실히 받아오기 (confirm 자동 처리)"""
    url = f"https://drive.google.com/uc?id={file_id}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(url, str(dst), quiet=False)

def ensure_model_ok(path: Path):
    """모델 파일이 정상인지 확인하고 문제면 재다운로드 후 검증. 실패 시 명확히 에러 표시."""
    # 처음이거나 손상 의심이면 다운
    if likely_broken(path):
        # 기존 찌꺼기 삭제
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass

        with st.spinner("📥 모델 파일을 Google Drive에서 다운로드 중..."):
            download_model_from_gdrive(GDRIVE_FILE_ID, path)

    # 다운받은 뒤에도 손상/HTML이면 중단
    if likely_broken(path):
        st.error(
            "모델 파일 다운로드가 올바르지 않습니다. (HTML/손상 감지)\n"
            "👉 Google Drive 공유가 **'링크가 있는 모든 사용자(보기)'** 인지 다시 확인해줘."
        )
        st.stop()

    # torch로 가볍게 열어보며 유효성 최종 체크 (메모리 큰 로드 아님)
    try:
        # ckpt 헤더만 파싱되는지 확인
        _ = torch.load(str(path), map_location="cpu")
        # 메모리 사용 줄이기 위해 즉시 deref
        del _
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except Exception as e:
        # 문제 있으면 파일 지우고 에러
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        st.error(f"모델 파일이 손상되었거나 호환되지 않습니다: {e}")
        st.stop()

# ==================== 모델 로드 ====================
@st.cache_resource
def load_model(path: Path):
    ensure_model_ok(path)
    return YOLO(str(path))

# ==================== 그리기/요약 ====================
def get_korean_font(size=18):
    """한글 폰트 로드: 프로젝트/fonts 우선, 없으면 OS 기본, 마지막엔 기본폰트"""
    candidates = [
        str(BASE_DIR / "fonts" / "NotoSansKR-Regular.ttf"),
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS
        "C:/Windows/Fonts/malgun.ttf",                 # Windows
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()

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

            # 박스
            draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=3)

            # 라벨(배경 + 텍스트)
            label = f"{cls_name} {conf:.2f}"
            try:
                # PIL>=9: textbbox
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
    conf_thres = st.slider("Confidence", 0.1, 0.9, 0.30, 0.05)
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
