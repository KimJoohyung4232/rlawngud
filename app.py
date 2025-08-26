# app.py
import io
from pathlib import Path

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
from ultralytics import YOLO

# ==================== 기본 설정 ====================
st.set_page_config(page_title="YOLO 탐지기", page_icon="🧠", layout="centered")

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "best.pt"   # 같은 폴더에 best.pt 두기!

DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 영어 → 한글 매핑 (필요 시 계속 추가)
KOR_LABELS = {
    # 짜장면 관련 다양한 스펠링 대비
    "jjajangmyeon": "짜장면",
    "jajangmyeon": "짜장면",
    "jjajang": "짜장면",
    "jajang": "짜장면",
    "blackbean_noodles": "짜장면",
    "blackbean": "짜장면",

    "ramen": "라면",
    "noodles": "면",
    # ...
}
def to_kor(name: str) -> str:
    # 소문자 normalize
    key = str(name).strip().lower()
    return KOR_LABELS.get(key, name)

# ==================== 유틸 ====================
@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        st.error(f"모델 파일이 없어요: {path}")
        st.stop()
    return YOLO(str(path))

def draw_boxes(pil_img: Image.Image, results, names_dict):
    """YOLO 결과 박스를 그려 PIL.Image로 반환 (라벨 한글화 포함)"""
    img = np.array(pil_img)[:, :, ::-1].copy()  # RGB->BGR (cv2)
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            conf = float(box.conf[0].cpu().numpy())
            cls  = int(box.cls[0].cpu().numpy())
            cls_eng = names_dict.get(cls, str(cls))
            cls_kor = to_kor(cls_eng)
            label = f"{cls_kor} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return Image.fromarray(img[:, :, ::-1])  # BGR->RGB

def summarize_prediction(rows):
    """탐지 rows를 바탕으로 클래스별 conf 합산 후 한 줄 요약 (한글)"""
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

            # 결과 이미지
            out_img = draw_boxes(st.session_state["uploaded_img"], results, names)
            st.session_state["pred_img"] = out_img

            # 표용 rows (라벨 한글화 저장)
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
                        "class_name": cls_kor,  # 한글
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

    # 한 줄 요약(한글)
    msg = st.session_state.get("summary_msg")
    if msg:
        st.info(msg)

    # ⚠ pandas/pyarrow 없이 안전하게 출력 (충돌 방지)
    if st.session_state.get("det_rows"):
        st.markdown("#### 탐지 박스 목록")
        st.json(st.session_state["det_rows"])  # ← 표 대신 JSON으로 출력
