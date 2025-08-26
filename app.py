# app.py
import io
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
# 릴리스에서 받은 파일명을 로컬에도 동일하게 저장 (default.pt)
MODEL_PATH = BASE_DIR / "default.pt"
# ✅ GitHub Releases Assets에 올라간 가중치의 직접 다운로드 URL
#    (네가 만든 first release 기준)
GITHUB_ASSET_URL = "https://github.com/KimJoohyung4232/rlawngud/releases/download/v1.0.0/default.pt"

DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==================== 라벨 한글 매핑 ====================
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


# ==================== 가중치 다운로드 유틸 ====================
def _looks_like_html(chunk: bytes) -> bool:
    head = chunk[:512].lower()
    return (b"<html" in head) or (b"<!doctype html" in head) or (b"{\"error" in head)

def download_weight_from_github(url: str, dst: Path):
    """GitHub Releases asset에서 YOLO 가중치 다운로드 (바이너리 검증 + 크기 검증 포함)"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(
        url, stream=True, headers={"Accept": "application/octet-stream"}, timeout=60
    ) as r:
        r.raise_for_status()
        first = True
        written = 0
        with open(dst, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if not chunk:
                    continue
                if first:
                    if _looks_like_html(chunk):
                        raise RuntimeError("가중치 대신 HTML/에러 페이지를 받았습니다. 릴리스 URL을 확인해 주세요.")
                    first = False
                f.write(chunk)
                written += len(chunk)
    # 너무 작으면 실패 처리 (대략 1MB 미만은 의심)
    if dst.stat().st_size < 1_000_000:
        dst.unlink(missing_ok=True)
        raise RuntimeError("가중치 파일 크기가 비정상적으로 작습니다. 릴리스 URL/파일을 확인해 주세요.")

@st.cache_resource(show_spinner=False)
def load_model(path: Path) -> YOLO:
    """가중치 다운로드(필요 시) + YOLO 로드. 손상/캐시 꼬임 시 1회 자동 재시도."""
    if not path.exists():
        with st.spinner("모델 다운로드 중... (GitHub Releases)"):
            download_weight_from_github(GITHUB_ASSET_URL, path)

    # 디바이스 표시용 간단 로그
    st.caption(f"Device: {DEVICE}")
    st.caption(f"Model file size: {path.stat().st_size:,} bytes")

    # 첫 로드
    try:
        return YOLO(str(path))
    except Exception as e:
        # 손상/캐시 꼬임 가능 → 강제 재다운 후 재시도
        st.warning("모델 로드 실패 🥲 가중치를 다시 받습니다…")
        path.unlink(missing_ok=True)
        with st.spinner("모델 재다운로드 중..."):
            download_weight_from_github(GITHUB_ASSET_URL, path)
        try:
            return YOLO(str(path))
        except Exception as e2:
            st.error(
                "모델 로드에 계속 실패했습니다.\n"
                f"원인: {type(e2).__name__}: {e2}\n\n"
                "➡️ Releases에 올라간 파일이 Ultralytics YOLO(v8/v11) 포맷의 .pt인지 확인해 주세요."
            )
            st.stop()


# ==================== 폰트 유틸 ====================
def get_korean_font(size=18):
    """한글 폰트 로드: 프로젝트 fonts 우선 → 시스템 기본"""
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

with st.sidebar:
    st.subheader("설정")
    conf_thres = st.slider("Confidence", 0.1, 0.9, 0.30, 0.05)
    iou_thres  = st.slider("IoU", 0.1, 0.9, 0.45, 0.05)
    st.write("모델 파일:", f"`{MODEL_PATH.name}`")
    # 응급 플랜 B: 모델 수동 업로드(발표 현장 대비)
    up_model = st.file_uploader("모델(.pt) 직접 업로드", type=["pt"])
    if up_model:
        data = up_model.read()
        MODEL_PATH.write_bytes(data)
        st.success(f"모델 교체 완료: {MODEL_PATH} ({len(data):,} bytes)")
        st.rerun()

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
