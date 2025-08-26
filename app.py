# app.py
import io
import re
from pathlib import Path
from typing import List, Dict

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import requests
from ultralytics import YOLO

# ==================== 기본 설정 ====================
st.set_page_config(page_title="YOLO 탐지기", page_icon="🧠", layout="centered")

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "default.pt"  # 릴리스에 올린 파일명과 동일
# ✅ GitHub Releases의 “Assets”에 뜨는 파일의 직접 다운로드 URL
MODEL_URL = "https://github.com/KimJoohyeong4232/rlawngud/releases/download/v1.0.0/default.pt"

DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 영어 → 한글 매핑(필요 시 자유롭게 추가)
KOR_LABELS: Dict[str, str] = {
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


# ==================== 모델 다운로드 유틸 ====================
def _http_download(url: str, dst: Path):
    """간단/견고한 HTTP 다운로드 (스트리밍, 재시도 포함)"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    ok = False
    last_err = None
    for _ in range(3):  # 최대 3회 재시도
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                # 내용물이 HTML(에러페이지)이 아닌지 간단 점검
                ctype = r.headers.get("content-type", "")
                if "text/html" in ctype.lower():
                    raise RuntimeError("다운로드된 응답이 HTML입니다. URL을 다시 확인하세요.")
                with open(dst, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)
            ok = True
            break
        except Exception as e:
            last_err = e
    if not ok:
        raise RuntimeError(f"모델 다운로드 실패: {last_err}")


@st.cache_resource(show_spinner=False)
def load_model(path: Path) -> YOLO:
    """로딩 + 필요시 GitHub 릴리스에서 모델 자동 다운로드"""
    if not path.exists() or path.stat().st_size < 10_000:  # 10KB 미만이면 손상으로 간주
        with st.spinner("모델 다운로드 중... (GitHub Releases)"):
            _http_download(MODEL_URL, path)
    # 여기서 바로 YOLO 로드 (UnpicklingError 방지: 완전한 .pt만 로드)
    return YOLO(str(path))


# ==================== 폰트 유틸 ====================
def get_korean_font(size=18):
    """한글 폰트 로드: 프로젝트 fonts 폴더 우선, 없으면 시스템 기본"""
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


# ==================== 박스 드로잉 ====================
def draw_boxes(pil_img: Image.Image, results, names: Dict[int, str], font=None):
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    font = font or get_korean_font(18)

    for r in results:
        if getattr(r, "boxes", None) is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0].item())
            cls  = int(box.cls[0].item())
            cls_eng = names.get(cls, str(cls))
            cls_name = to_kor(cls_eng)

            # 박스
            draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=3)

            # 라벨
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


def summarize_prediction(rows: List[Dict]) -> str:
    if not rows:
        return "아직 확신하기 어려워요. (탐지 결과 없음)"
    totals: Dict[str, float] = {}
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
    st.write("모델 파일:", f"`{MODEL_PATH.name}`")
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

# ==================== 모델 로드 ====================
model = load_model(MODEL_PATH)

# ==================== 추론 ====================
if run_btn:
    if "uploaded_img" not in st.session_state:
        st.warning("먼저 이미지를 올려줘!")
    else:
        with st.spinner("모델 추론 중..."):
            # Streamlit Cloud는 GPU가 없을 수 있으므로 안전하게 선택
            dv = "mps" if DEVICE == "mps" else (0 if DEVICE == "cuda" else "cpu")
            img_np = np.array(st.session_state["uploaded_img"])
            results = model.predict(
                source=img_np, conf=conf_thres, iou=iou_thres,
                verbose=False, device=dv
            )
            names = model.names

            out_img = draw_boxes(st.session_state["uploaded_img"], results, names)
            st.session_state["pred_img"] = out_img

            rows: List[Dict] = []
            for r in results:
                if getattr(r, "boxes", None) is None:
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
