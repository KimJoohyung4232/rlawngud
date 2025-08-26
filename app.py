# app.py
import io
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from ultralytics import YOLO

# ==================== 기본 설정 ====================
st.set_page_config(page_title="YOLO 탐지기", page_icon="🧠", layout="centered")

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "best.pt"   # 같은 폴더에 best.pt 두기!

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
    key = str(name).strip().lower()
    return KOR_LABELS.get(key, name)

# ==================== 유틸 ====================
@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        st.error(f"모델 파일이 없어요: {path}")
        st.stop()
    return YOLO(str(path))

def draw_boxes(pil_img: Image.Image, results, names_dict, use_korean=True, font_path=None):
    """
    YOLO 결과를 PIL로 그려서 반환.
    - use_korean: True면 한글 매핑 사용(이미지에 한글 라벨 표시 시 폰트 필요)
    - font_path: NotoSansKR 같은 TTF 경로. 없으면 기본 폰트(한글 미표시 가능)
    """
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)

    # 폰트 준비
    if font_path is not None:
        try:
            font = ImageFont.truetype(font_path, size=18)
        except Exception:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0].item())
            cls  = int(box.cls[0].item())
            cls_eng = names_dict.get(cls, str(cls))
            cls_name = to_kor(cls_eng) if use_korean else cls_eng

            # 박스
            draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=3)

            # 라벨 배경 + 텍스트
            label = f"{cls_name} {conf:.2f}"
            tw, th = draw.textbbox((0, 0), label, font=font)[2:]
            pad = 4
            bx2 = x1 + tw + pad * 2
            by2 = y1 - th - pad * 2
            if by2 < 0:
                by2 = y1 + th + pad * 2  # 위에 못 그리면 박스 안쪽/아래쪽으로
                ty = y1 + pad
            else:
                ty = y1 - th - pad
            draw.rectangle([(x1, y1), (bx2, by2)], fill=(0, 255, 0))
            draw.text((x1 + pad, ty), label, font=font, fill=(0, 0, 0))

    return img

def summarize_prediction(rows):
    if not rows:
        return "아직 확신하기 어려워요. (탐지 결과 없음)"
    totals = {}
    for r in rows:
        name = r["class_name"]
        totals[name] = totals.get(name, 0.0) + float(r["conf"])
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
            img_np = np.array(st.session_state["uploaded_img"])  # RGB np.array
            results = model.predict(
                source=img_np, conf=conf_thres, iou=iou_thres,
                verbose=False, device=dv
            )
            names = model.names  # {idx: "class_name"}

            # 결과 이미지 (폰트가 있으면 font_path에 경로 넣어줘)
            out_img = draw_boxes(
                st.session_state["uploaded_img"],
                results, names,
                use_korean=True,
                font_path=None  # 예: str(BASE_DIR / "NotoSansKR-Regular.ttf")
            )
            st.session_state["pred_img"] = out_img

            # 표용 rows
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
