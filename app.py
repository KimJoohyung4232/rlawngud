# app.py
import io
import re
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
MODEL_PATH = BASE_DIR / "best.pt"   # 모델 캐싱 파일명

# ✅ 주형이 드라이브에 올린 best.pt 파일 ID
GDRIVE_FILE_ID = "1DsRNTxESZM5LTEWuV-QgezYkQ386WcTp"

DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==================== 라벨 한글 매핑 ====================
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

# ==================== 구글드라이브 다운로드 유틸 ====================
def _gdrive_confirm_token(resp):
    # 큰 파일일 때 등장하는 경고 토큰 잡아오기
    for k, v in resp.cookies.items():
        if k.startswith("download_warning"):
            return v
    m = re.search(r"confirm=([0-9A-Za-z_]+)&", resp.text)
    return m.group(1) if m else None

def download_from_gdrive(file_id: str, dst: Path):
    URL = "https://drive.google.com/uc?export=download"
    with requests.Session() as s:
        r = s.get(URL, params={"id": file_id}, stream=True)
        token = _gdrive_confirm_token(r)
        if token:
            r = s.get(URL, params={"id": file_id, "confirm": token}, stream=True)
        r.raise_for_status()
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)

# ==================== 폰트 유틸 (윈/맥/리눅스 호환) ====================
def get_korean_font(size=18):
    candidates = [
        str(BASE_DIR / "fonts" / "NotoSansKR-Regular.ttf"),  # 프로젝트 동봉
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",        # macOS
        "C:/Windows/Fonts/malgun.ttf",                       # Windows
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",  # 리눅스
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()

# ==================== 모델 로드 (캐시) ====================
@st.cache_resource
def load_model(path: Path):
    if not path.exists() or path.stat().st_size < 10_000_000:  # 10MB 미만이면 재다운로드 가정
        with st.spinner("모델 다운로드 중... (Google Drive)"):
            download_from_gdrive(GDRIVE_FILE_ID, path)
    # YOLO가 내부적으로 torch.load를 쓰는데, 배포 환경은 CPU 기준이라 map_location은 내부에서 처리됨
    return YOLO(str(path))

# ==================== 박스 그리기 ====================
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

            # 라벨
            label = f"{cls_name} {conf:.2f}"
            try:
                # textbbox: (left, top, right, bottom)
                l, t, r, b = draw.textbbox((0, 0), label, font=font)
                tw, th = r - l, b - t
            except Exception:
                tw, th = font.getsize(label)

            pad = 4
            if y1 - th - pad * 2 < 0:
                # 위에 공간 없으면 박스 안으로
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

def human_filesize(n):
    for unit in ["B","KB","MB","GB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"

# ==================== UI ====================
st.title("🧠 YOLO 객체 탐지 (Streamlit)")
st.caption(f"Device: {DEVICE}")

with st.sidebar:
    st.subheader("설정")
    conf_thres = st.slider("Confidence", 0.1, 0.9, 0.25, 0.05)
    iou_thres  = st.slider("IoU", 0.1, 0.9, 0.45, 0.05)
    size_txt = human_filesize(MODEL_PATH.stat().st_size) if MODEL_PATH.exists() else "없음"
    st.write("모델 파일:", f"`{MODEL_PATH.name}` ({size_txt})")
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
            img_np = np.array(st.session_state["uploaded_img"])  # RGB ndarray
            results = model.predict(
                source=img_np, conf=conf_thres, iou=iou_thres,
                verbose=False, device=dv
            )
            names = model.names  # {idx: class_name}

            # 결과 이미지
            out_img = draw_boxes(st.session_state["uploaded_img"], results, names)
            st.session_state["pred_img"] = out_img

            # rows(표/JSON용)
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
