# app.py
import io
import os
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
MODEL_PATH = BASE_DIR / "default.pt"

# 너 깃헙 릴리스에 올린 .pt 주소 (필요시 바꿔)
GITHUB_ASSET_URL = (
    "https://github.com/KimJoohyung4232/rlawngud/releases/download/v1.0.0/default.pt"
)

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
    return (b"<html" in head) or (b"<!doctype html" in head) or (b'{"error' in head)

def download_weight_from_github(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(
        url, stream=True, headers={"Accept": "application/octet-stream"}, timeout=60
    ) as r:
        r.raise_for_status()
        first = True
        with open(dst, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if not chunk:
                    continue
                if first:
                    if _looks_like_html(chunk):
                        raise RuntimeError("가중치 대신 HTML/에러 페이지를 받았습니다. 릴리스 URL을 확인해 주세요.")
                    first = False
                f.write(chunk)
    if dst.stat().st_size < 1_000_000:
        dst.unlink(missing_ok=True)
        raise RuntimeError("가중치 파일 크기가 비정상적으로 작습니다. 릴리스 URL/파일을 확인해 주세요.")

@st.cache_resource(show_spinner=False)
def load_model(path: Path) -> YOLO:
    """가중치가 없으면 깃헙 릴리스에서 받고, YOLO 모델 로드"""
    if not path.exists():
        with st.spinner("모델 다운로드 중... (GitHub Releases)"):
            download_weight_from_github(GITHUB_ASSET_URL, path)

    st.caption(f"Device: {DEVICE}")
    st.caption(f"Model file size: {path.stat().st_size:,} bytes")

    try:
        return YOLO(str(path))
    except Exception:
        st.warning("모델 로드 실패 😵 가중치를 다시 다운로드합니다…")
        path.unlink(missing_ok=True)
        with st.spinner("모델 재다운로드 중..."):
            download_weight_from_github(GITHUB_ASSET_URL, path)
        return YOLO(str(path))

# ==================== 폰트 유틸 (맥/윈도/리눅스/클라우드 모두 OK) ====================
def get_korean_font(size=20):
    """
    1) 프로젝트 fonts/에 TTF/OTF 있으면 사용
    2) 시스템 한글 폰트(맥: AppleSDGothicNeo, 윈도: 맑은고딕, 리눅스: 나눔/노토)
    3) 아무것도 없으면 NotoSansKR 자동 다운로드 후 사용
    """
    fonts_dir = BASE_DIR / "fonts"
    fonts_dir.mkdir(exist_ok=True)

    local_candidates = [
        fonts_dir / "NotoSansKR-Regular.ttf",
        fonts_dir / "NotoSansKR-Regular.otf",
    ]
    system_candidates = [
        # macOS
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        # Windows
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/malgunbd.ttf",
        # Linux (일반)
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKkr-Regular.otf",
    ]

    # 1) 로컬 동봉 폰트
    for p in local_candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception:
                pass

    # 2) 시스템 폰트
    for p in system_candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue

    # 3) 자동 다운로드 (한 번만)
    try:
        url_list = [
            # 구글 노토 산스 KR (OTF)
            "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/Korean/NotoSansKR-Regular.otf",
            # 네이버 나눔고딕 대체
            "https://github.com/naver/nanumfont/raw/master/LICENSE/NanumGothic.ttf",  # if first fails
        ]
        for url in url_list:
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                suffix = ".otf" if url.lower().endswith(".otf") else ".ttf"
                out = fonts_dir / f"auto-font{suffix}"
                out.write_bytes(r.content)
                return ImageFont.truetype(str(out), size)
            except Exception:
                continue
    except Exception:
        pass

    # 4) 최후의 수단(영문 전용)
    return ImageFont.load_default()

# ==================== 박스 드로잉 ====================
def draw_boxes(pil_img: Image.Image, results, names: Dict[int, str], font=None):
    img = pil_img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    font = font or get_korean_font(20)

    for r in results:
        if getattr(r, "boxes", None) is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0].item())
            cls  = int(box.cls[0].item())
            cls_eng = names.get(cls, str(cls))
            cls_name = to_kor(cls_eng)

            # 사각형
            draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=3)

            # 라벨 텍스트
            label = f"{cls_name} {conf:.2f}"
            try:
                left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
                tw, th = right - left, bottom - top
            except Exception:
                tw, th = font.getsize(label)

            pad = 6
            rx1 = x1
            ry2 = y1
            ry1 = y1 - th - pad * 2
            if ry1 < 0:  # 위쪽 공간 없으면 박스 안쪽으로
                ry1 = y1
                ry2 = y1 + th + pad * 2
            rx2 = x1 + tw + pad * 2

            # 둥근 배경
            try:
                draw.rounded_rectangle([(rx1, ry1), (rx2, ry2)], radius=6, fill=(0, 255, 0))
            except Exception:
                draw.rectangle([(rx1, ry1), (rx2, ry2)], fill=(0, 255, 0))

            draw.text((rx1 + pad, ry1 + pad), label, font=font, fill=(0, 0, 0))

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

    # 발표 현장 대비: 가중치 수동 업로드
    up_model = st.file_uploader("모델(.pt) 직접 업로드", type=["pt"])
    if up_model:
        data = up_model.read()
        MODEL_PATH.write_bytes(data)
        st.success(f"모델 교체 완료: {MODEL_PATH} ({len(data):,} bytes)")
        st.rerun()

    # 디버그 JSON 토글 (기본 꺼짐)
    show_debug = st.checkbox("디버그 JSON 보기", value=False)

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
            st.session_state["det_rows"]   = rows
            st.session_state["summary_msg"] = summarize_prediction(rows)

        st.success("예측 완료!")

# ==================== 결과 표시 ====================
if "pred_img" in st.session_state:
    st.markdown("### 3) 결과")
    st.image(st.session_state["pred_img"], caption="탐지 결과", use_container_width=True)

    msg = st.session_state.get("summary_msg")
    if msg:
        st.info(msg)

    # 디버그 모드일 때만 JSON 출력
    if show_debug and st.session_state.get("det_rows"):
        st.markdown("#### 탐지 박스 목록 (디버그)")
        st.json(st.session_state["det_rows"])

    # 다운로드 버튼
    buf = io.BytesIO()
    st.session_state["pred_img"].save(buf, format="PNG")
    st.download_button(
        "📥 결과 이미지 저장", data=buf.getvalue(),
        file_name="prediction.png", mime="image/png", use_container_width=True
    )
