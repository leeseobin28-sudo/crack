import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np
import cv2
import os

# 아이폰 HEIC/HEIF 지원
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:
    HEIC_SUPPORTED = False

# ── 기준 물체: 신용카드/체크카드/학생증 (ISO/IEC 7810 ID-1 규격) ──
CARD_W_MM = 85.60
CARD_H_MM = 53.98
CARD_AREA_MM2 = CARD_W_MM * CARD_H_MM   # 4620.7 mm² ≈ 46.21 cm²
CARD_RATIO = CARD_W_MM / CARD_H_MM      # ≈ 1.586

st.set_page_config(page_title="균열 실제 면적 탐지 (카드 기준)", page_icon="🔍", layout="wide")
st.title("🔍 콘크리트 균열 실제 면적 탐지 앱 — 카드 기준 스케일")
st.write("균열 옆 **같은 벽면에 카드(신용카드·학생증 규격)를 붙이고** 촬영하면, "
         "카드의 네 모서리로 원근을 보정해 균열의 **실제 면적(mm², cm²)**을 계산합니다.")

if not HEIC_SUPPORTED:
    st.warning("💡 HEIC 파일을 올리려면 `pip install pillow-heif`를 설치해주세요.")

# 모델은 이 코드 파일과 같은 폴더에서 찾음 → 내 컴퓨터와 Streamlit Cloud 서버 모두에서 작동
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CRACK_MODEL_PATH = os.path.join(BASE_DIR, "bestcrack.pt")
CARD_MODEL_PATH = os.path.join(BASE_DIR, "best.pt")         # 카드 pose 모델 (없으면 OpenCV로 자동 검출)


@st.cache_resource
def load_model(path):
    return YOLO(path)


try:
    model_crack = load_model(CRACK_MODEL_PATH)
    st.success(f"✅ 균열 모델 로드 완료 ({CRACK_MODEL_PATH})")
except Exception as e:
    st.error(f"❌ 균열 모델 로드 실패: {e}")
    st.stop()

model_card = None
if os.path.exists(CARD_MODEL_PATH):
    try:
        model_card = load_model(CARD_MODEL_PATH)
        st.success(f"✅ 카드 모델 로드 완료 ({CARD_MODEL_PATH})")
    except Exception as e:
        st.warning(f"카드 모델 로드 실패 → OpenCV 자동 검출 사용: {e}")
else:
    st.info("ℹ️ 카드 모델(best.pt)이 없어 OpenCV 사각형 검출로 카드를 찾습니다.")


# ───────────────────────── 카드 검출 ─────────────────────────
def order_corners(pts):
    """4점을 좌상단부터 시계방향으로 정렬 (카드가 기울어져 있어도 안정적)"""
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0).astype(np.float32)


def polygon_to_quad(poly):
    """마스크 윤곽(모서리가 둥근 카드 포함)을 4개 꼭짓점으로 근사"""
    poly = np.array(poly, dtype=np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(poly)
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.08, 15):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return order_corners(approx)
    return order_corners(cv2.boxPoints(cv2.minAreaRect(hull)))


def quad_aspect(q):
    w = (np.linalg.norm(q[0] - q[1]) + np.linalg.norm(q[3] - q[2])) / 2
    h = (np.linalg.norm(q[0] - q[3]) + np.linalg.norm(q[1] - q[2])) / 2
    return max(w, h) / max(min(w, h), 1e-6)


def detect_card_yolo(img, conf, kpt_conf=0.5):
    """카드 pose 모델(키포인트 4개 = 카드 네 모서리)로 카드 검출.
    반환: (꼭짓점 4개 또는 None, 박스 꼭짓점 또는 None)"""
    res = model_card.predict(source=img, conf=conf, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None, None
    i = int(np.argmax(res.boxes.conf.cpu().numpy()))
    x1, y1, x2, y2 = res.boxes.xyxy[i].cpu().numpy()
    box_q = order_corners([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

    # 1순위: 키포인트(모서리 4점) → 원근 보정 가능
    if res.keypoints is not None and len(res.keypoints.xy) > i:
        kp = res.keypoints.xy[i].cpu().numpy()
        kc = res.keypoints.conf
        kc = kc[i].cpu().numpy() if kc is not None else np.ones(len(kp))
        if len(kp) == 4 and (kc >= kpt_conf).all():
            q = order_corners(kp)
            if cv2.isContourConvex(q.reshape(-1, 1, 2)) and abs(quad_aspect(q) - CARD_RATIO) < 0.6:
                return q, box_q
    # 2순위: 세그멘테이션 모델이면 마스크 윤곽 사용
    if res.masks is not None and len(res.masks.xy[i]) >= 4:
        return polygon_to_quad(res.masks.xy[i]), box_q
    return None, box_q


def detect_card_opencv(img):
    """카드 비율(1.586)에 가까운 볼록 사각형 중 가장 그럴듯한 것 선택"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img_area = img.shape[0] * img.shape[1]
    best, best_score = None, 1e9
    for c in contours:
        a = cv2.contourArea(c)
        if a < 0.003 * img_area or a > 0.6 * img_area:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        q = order_corners(approx)
        err = abs(quad_aspect(q) - CARD_RATIO)
        if err < 0.25:
            score = err - 1e-9 * a            # 비율 우선, 비슷하면 큰 것
            if score < best_score:
                best, best_score = q, score
    return best


def _card_rects(m, img_area, k):
    """이진 마스크에서 '카드처럼 생긴 직사각형'(비율 1.586, 꽉 찬 모양) 후보들을 점수와 함께 반환"""
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k * 3, k * 3), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 0.003 * img_area or a > 0.5 * img_area:
            continue
        rect = cv2.minAreaRect(cv2.convexHull(c))
        w, h = rect[1]
        if min(w, h) < 1:
            continue
        asp = max(w, h) / min(w, h)
        fill = a / (w * h)                 # 직사각형에 얼마나 꽉 차는지
        if fill < 0.75 or abs(asp - CARD_RATIO) > 0.35:
            continue
        out.append((abs(asp - CARD_RATIO) + (1 - fill), order_corners(cv2.boxPoints(rect))))
    return out


def detect_card_contrast(img):
    """배경(콘크리트)과 '다르게 보이는' 직사각형을 카드 후보로 찾음.
    ① 채도가 높다 → 유채색 카드(파랑·노랑 등)
    ② 배경보다 훨씬 어둡다 → 검정 카드
    ③ 배경보다 훨씬 밝다 → 흰색 카드
    반환: [(꼭짓점 4개, 검출 종류), ...]  (그럴듯한 순서, 같은 물체 중복 제거)"""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    S = hsv[..., 1]
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (5, 5), 0).astype(np.int16)
    low_sat = S < 60
    bg = float(np.median(gray[low_sat])) if low_sat.mean() > 0.2 else float(np.median(gray))  # 벽 밝기
    k = max(3, (min(img.shape[:2]) // 150) | 1)
    img_area = img.shape[0] * img.shape[1]

    tests = [("유채색 카드(채도)", [(S > t) for t in (170, 140, 110, 80)]),
             ("검정 카드(배경보다 어두움)", [(gray < bg - d) & (S < 120) for d in (70, 50, 35)]),
             ("흰색 카드(배경보다 밝음)", [(gray > bg + d) & (S < 60) for d in (50, 35, 22)])]
    found = []
    for kind, masks in tests:
        for m in masks:                     # 엄격한 기준 → 느슨한 기준 순서
            for score, q in sorted(_card_rects(m.astype(np.uint8), img_area, k), key=lambda x: x[0]):
                c = q.mean(axis=0)
                size = np.linalg.norm(q[0] - q[2])
                if all(np.linalg.norm(c - f[0].mean(axis=0)) > 0.3 * size for f in found):
                    found.append((q, kind))
    return found


def _fit_line_robust(pts, iters=4):
    """이상점(손가락 경계 등)을 반복적으로 버리면서 직선 맞춤"""
    pts = np.asarray(pts, np.float32)
    keep = np.ones(len(pts), bool)
    for _ in range(iters):
        if keep.sum() < 6:
            return None
        vx, vy, x0, y0 = cv2.fitLine(pts[keep], cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        r = np.abs((pts - [x0, y0]) @ np.array([-vy, vx]))
        keep = r < max(1.5, 2.5 * np.median(r[keep]))
    return np.array([x0, y0]), np.array([vx, vy])


def _intersect(l1, l2):
    (p1, d1), (p2, d2) = l1, l2
    A = np.array([d1, -d2]).T
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    t = np.linalg.solve(A, p2 - p1)
    return p1 + t[0] * d1


def refine_quad(img, q, band=0.08, n=80):
    """카드의 네 '변'을 각각 직선으로 다시 맞추고, 인접한 두 직선의 교점을 모서리로 사용.
    → 손가락에 모서리가 가려져도 변의 보이는 부분만으로 가려진 모서리를 복원.
    → 카드의 둥근 모서리도 규격(85.6×53.98, 둥글리기 전 직사각형 기준)에 맞는 가상 꼭짓점이 됨."""
    q = order_corners(q)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sat = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[..., 1].astype(np.float32)
    feat = cv2.GaussianBlur(0.5 * gray + sat, (5, 5), 0)   # 채도 포함 → 벽 그림자 경계에 덜 속음
    center = q.mean(axis=0)
    lines = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        L = np.linalg.norm(b - a)
        d = (b - a) / max(L, 1e-6)
        nrm = np.array([-d[1], d[0]])
        if np.dot(center - (a + b) / 2, nrm) > 0:          # 법선을 카드 바깥쪽으로
            nrm = -nrm
        w = max(4.0, band * L)
        offs = np.linspace(-w, w, int(2 * w) + 1)
        base = a[None] + np.linspace(0.06, 0.94, n)[:, None] * (b - a)[None]
        xs = (base[:, 0:1] + offs[None] * nrm[0]).astype(np.float32)
        ys = (base[:, 1:2] + offs[None] * nrm[1]).astype(np.float32)
        prof = cv2.remap(feat, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        g = np.diff(prof, axis=1)                              # 안 → 밖 방향 변화량
        # 한 변 전체에서 변화 부호는 같아야 함(다수결) → 반대 부호 경계(손가락 등) 무시
        sign = np.sign(np.median(g[np.arange(n), np.argmax(np.abs(g), axis=1)]))
        gg = np.clip(g * sign, 0, None) if sign != 0 else np.abs(g)
        j = np.argmax(gg, axis=1)
        strength = gg[np.arange(n), j]
        ok = strength > max(4.0, 0.3 * np.median(strength))
        pts = (base + ((offs[j] + offs[j + 1]) / 2)[:, None] * nrm[None])[ok]
        fit = _fit_line_robust(pts) if len(pts) >= 6 else None
        lines.append((a, d) if fit is None else fit)
    new = []
    for i in range(4):
        c = _intersect(lines[(i - 1) % 4], lines[i])
        new.append(q[i] if c is None else c)
    new = np.array(new, np.float32)
    # 안전장치: 너무 많이 움직였거나 모양이 이상하면 보정 전 값 유지
    diag = np.linalg.norm(q[0] - q[2])
    if (np.linalg.norm(new - q, axis=1).max() > 0.25 * diag
            or not cv2.isContourConvex(new.reshape(-1, 1, 2))
            or abs(quad_aspect(new) - CARD_RATIO) > 0.5):
        return q, False
    return new, True


def edge_map(img):
    """경계선 세기 지도 (밝기+채도 변화량). 위치 오차 ±2px 허용"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sat = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[..., 1].astype(np.float32)
    feat = cv2.GaussianBlur(0.5 * gray + sat, (5, 5), 0)
    mag = cv2.magnitude(cv2.Sobel(feat, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(feat, cv2.CV_32F, 0, 1, ksize=3))
    return cv2.dilate(mag, np.ones((5, 5), np.uint8))


def edge_score(emap, q, n=60):
    """사각형의 네 변이 실제 '경계선' 위에 있는지 점수화 (네 변 중 가장 약한 변의 경계 세기).
    잘못 잡은 변(카드 안쪽을 가로지르는 선)은 경계가 없어 점수가 낮음."""
    q = order_corners(q)
    scores = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        pts = (a[None] + np.linspace(0.1, 0.9, n)[:, None] * (b - a)[None]).astype(np.float32)
        v = cv2.remap(emap, pts[:, 0:1], pts[:, 1:2], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        scores.append(float(np.median(v)))
    return min(scores)


def quad_angles(q):
    """사각형 네 내각(도)"""
    out = []
    for i in range(4):
        u, v = q[i - 1] - q[i], q[(i + 1) % 4] - q[i]
        c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
        out.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
    return np.array(out)


def is_card_shape(q, max_angle_dev=15, min_side_ratio=0.85, ratio_tol=0.25):
    """카드(직사각형)를 사진으로 찍었을 때 나올 수 있는 모양인지 검사.
    원근 때문에 약간 찌그러지는 건 허용하지만, 내각이 90°에서 15° 넘게 벗어나거나
    마주 보는 변 길이가 15% 넘게 차이 나면 '카드가 아닌 모양'으로 판단"""
    if not cv2.isContourConvex(q.reshape(-1, 1, 2).astype(np.float32)):
        return False
    if np.abs(quad_angles(q) - 90).max() > max_angle_dev:
        return False
    L = [np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]
    if min(L[0], L[2]) / max(L[0], L[2]) < min_side_ratio or min(L[1], L[3]) / max(L[1], L[3]) < min_side_ratio:
        return False
    return abs(quad_aspect(q) - CARD_RATIO) <= ratio_tol


def _rect_quad(cx, cy, w, th):
    h = w / CARD_RATIO
    c, s = np.cos(th), np.sin(th)
    pts = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    return order_corners(pts @ np.array([[c, -s], [s, c]]).T + [cx, cy])


def fit_card_rect(emap, q0):
    """가로세로비를 카드 규격(1.586)으로 고정한 직사각형을 놓고, 네 변이 실제 경계선 위에 오도록
    위치·크기·회전을 조금씩 바꿔가며 찾음 → 결과는 항상 '카드 모양'"""
    (cx, cy), (w, h), ang = cv2.minAreaRect(q0.reshape(-1, 1, 2).astype(np.float32))
    th = np.radians(ang) if w >= h else np.radians(ang + 90)
    long0 = max(w, h, 10)
    best = None
    for sc in (0.85, 1.0, 1.15, 1.3):                 # 처음 크기가 틀렸을 수 있어 여러 크기에서 출발
        p = np.array([cx, cy, long0 * sc, th])
        s = edge_score(emap, _rect_quad(*p))
        steps = np.array([0.05 * long0, 0.05 * long0, 0.04 * long0, np.radians(3)])
        for _ in range(60):
            improved = False
            for k in range(4):
                for sgn in (1, -1):
                    t = p.copy()
                    t[k] += sgn * steps[k]
                    st_ = edge_score(emap, _rect_quad(*t))
                    if st_ > s:
                        p, s, improved = t, st_, True
            if not improved:
                steps /= 2
                if steps[0] < 0.5:
                    break
        if best is None or s > best[1]:
            best = (_rect_quad(*p), s)
    return best[0]


def make_card_shape(img, emap, q):
    """최종 카드 사각형이 카드 모양이 아니면, 카드 비율 직사각형으로 다시 맞춤.
    반환: (사각형, 다시 맞췄는지 여부)"""
    if is_card_shape(q):
        return q, False
    rq = fit_card_rect(emap, q)
    q2, ok = refine_quad(img, rq)                     # 약간의 원근(사다리꼴)은 다시 허용
    return (q2 if ok and is_card_shape(q2) else rq), True


def _overlaps(q1, q2):
    """두 사각형이 같은 물체를 가리키는지 (한쪽 중심이 다른 쪽 안에 있음)"""
    c1, c2 = q1.mean(axis=0), q2.mean(axis=0)
    return (cv2.pointPolygonTest(q2.reshape(-1, 1, 2), (float(c1[0]), float(c1[1])), False) >= 0 or
            cv2.pointPolygonTest(q1.reshape(-1, 1, 2), (float(c2[0]), float(c2[1])), False) >= 0)


def card_homography(q):
    """이미지 픽셀 → 벽면 실제 좌표(mm) 변환 행렬"""
    top = np.linalg.norm(q[0] - q[1])
    left = np.linalg.norm(q[0] - q[3])
    if top >= left:   # 카드가 가로로 놓임
        dst = np.array([[0, 0], [CARD_W_MM, 0], [CARD_W_MM, CARD_H_MM], [0, CARD_H_MM]], np.float32)
    else:             # 카드가 세로로 놓임
        dst = np.array([[0, 0], [CARD_H_MM, 0], [CARD_H_MM, CARD_W_MM], [0, CARD_W_MM]], np.float32)
    return cv2.getPerspectiveTransform(q, dst)


# ───────────────────────── 균열 마스크 & 면적 ─────────────────────────
def raster_mask(res, shape, card_q=None):
    """YOLO 마스크를 '다각형'이 아니라 '픽셀 그대로' 합침.
    (masks.xy는 조각난 마스크를 한 다각형으로 이어 붙여서 가느다란 선 모양 오류가 생김)"""
    H, W = shape[:2]
    m = np.zeros((H, W), np.uint8)
    if res.masks is not None:
        for x in res.masks.data.cpu().numpy():
            if x.shape != (H, W):
                x = cv2.resize(x, (W, H), interpolation=cv2.INTER_LINEAR)
            m |= (x > 0.5).astype(np.uint8)
    if card_q is not None:
        cv2.fillPoly(m, [card_q.astype(np.int32)], 0)
    return m


def refine_crack(img, mask, card_q=None):
    """2단계 검출: YOLO는 '균열이 있는 곳'만 찾고, 실제 균열 픽셀은
    '주변보다 어두운 가느다란 선'(Black-hat 변환)으로 다시 골라냄.
    → 두껍게 칠해진 마스크가 실제 균열 폭으로 줄어들고, 손·벽돌처럼 넓고 매끈한 오탐은 대부분 빠짐."""
    if mask.sum() == 0:
        return mask
    H, W = mask.shape
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    k = max(9, (min(H, W) // 60) | 1)                    # 균열 최대 폭보다 큰 커널
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    region = cv2.dilate(mask, np.ones((5, 5), np.uint8))  # 마스크 경계에 걸친 균열도 포함
    vals = bh[region > 0].reshape(-1, 1)
    t = max(15.0, cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
    fine = ((bh > t) & (region > 0)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fine, 8)
    min_px = max(20, int(H * W * 2e-5))                   # 자잘한 점(기공·얼룩) 제거
    keep_ids = np.where(stats[1:, cv2.CC_STAT_AREA] >= min_px)[0] + 1
    out = np.isin(lab, keep_ids).astype(np.uint8)
    if card_q is not None:
        cv2.fillPoly(out, [card_q.astype(np.int32)], 0)
    return out


def metric_area_mm2(mask, H, max_side=6000):
    """균열 마스크를 벽면 평면(mm)으로 투영한 뒤 면적 계산 → 원근 왜곡 보정"""
    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    mm_contours = [cv2.perspectiveTransform(c.astype(np.float32), H) for c in contours]
    allpts = np.concatenate([c.reshape(-1, 2) for c in mm_contours])
    xmin, ymin = allpts.min(axis=0)
    xmax, ymax = allpts.max(axis=0)
    res_mm = max(0.05, max(xmax - xmin, ymax - ymin) / max_side)   # 캔버스 1px당 mm
    Wc = int(np.ceil((xmax - xmin) / res_mm)) + 2
    Hc = int(np.ceil((ymax - ymin) / res_mm)) + 2
    canvas = np.zeros((Hc, Wc), np.uint8)
    shifted = [((c.reshape(-1, 2) - [xmin, ymin]) / res_mm).astype(np.int32) for c in mm_contours]
    holes = [hier[0][i][3] != -1 for i in range(len(shifted))]
    cv2.fillPoly(canvas, [c for c, h in zip(shifted, holes) if not h], 1)   # 바깥 윤곽 채우기
    inner = [c for c, h in zip(shifted, holes) if h]
    if inner:
        cv2.fillPoly(canvas, inner, 0)                                     # 구멍 비우기
    return float(canvas.sum()) * res_mm ** 2


# ───────────────────────── 그리기 ─────────────────────────
def draw(img, crack_mask, card_q):
    out = img.copy()
    overlay = img.copy()
    overlay[crack_mask > 0] = (255, 0, 0)
    cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)
    cnts, _ = cv2.findContours(crack_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (255, 0, 0), 2)
    if card_q is not None:
        t = max(2, img.shape[1] // 400)
        cv2.polylines(out, [card_q.astype(np.int32)], True, (0, 200, 255), t)
        for p in card_q:
            cv2.circle(out, tuple(p.astype(int)), t * 3, (0, 200, 255), -1)
    return out


# ───────────────────────── UI ─────────────────────────
allowed = ["jpg", "jpeg", "png"] + (["heic", "heif"] if HEIC_SUPPORTED else [])
st.markdown("---")
up = st.file_uploader(f"📸 카드와 균열이 함께 찍힌 사진 ({', '.join(allowed).upper()})", type=allowed)

with st.expander("📷 촬영 가이드", expanded=False):
    st.markdown(
        "- 카드는 **균열과 같은 평면(벽면)에 밀착**시켜 주세요. 떠 있으면 오차가 커집니다.\n"
        "- 카드는 **손으로 들지 말고 테이프로 붙이면** 손을 균열로 오인하는 문제가 없어집니다.\n"
        "- 카드는 **균열 가까이**, 네 모서리가 모두 보이게 찍어주세요.\n"
        "- 카드가 화면에서 클수록(너무 멀리서 찍지 않을수록) 정확합니다.\n"
        "- 비스듬히 찍어도 원근 보정되지만, 가능하면 정면에 가깝게 찍는 게 좋습니다.\n"
        "- 벽과 색이 비슷한 카드는 잘 안 잡힙니다. **흰 벽에는 검정·유채색 카드**, 어두운 벽에는 흰색 카드를 쓰세요.")

if up is not None:
    image = ImageOps.exif_transpose(Image.open(up))
    img = np.array(image.convert("RGB"))

    c1, c2 = st.columns(2)
    with c1:
        crack_conf = st.slider("균열 탐지 민감도", 0.05, 0.95, 0.25, 0.05)
    with c2:
        card_conf = st.slider("카드 탐지 민감도", 0.05, 0.95, 0.40, 0.05, disabled=model_card is None)
    use_refine = st.checkbox("카드 변 직선 보정 (손가락에 가려진 모서리 복원)", value=True)
    use_fine = st.checkbox("균열 픽셀 정밀화 (YOLO 영역 안에서 어두운 선만 남기기)", value=True)
    card_pick = st.number_input("카드 후보 번호 (카드가 아닌 물체를 잡았을 때 2, 3…으로 바꿔 다시 측정)",
                                min_value=1, max_value=10, value=1, step=1)

    if st.button("🚀 정밀 측정 시작", type="primary", use_container_width=True):
        with st.spinner("카드 검출 및 균열 면적 계산 중..."):
            card_q, box_q = detect_card_yolo(img, card_conf) if model_card else (None, None)
            method = "YOLO 카드 모델 (모서리 키포인트)"
            cands = detect_card_contrast(img)              # 유채색·검정·흰색 카드 후보
            if card_q is not None and use_refine:
                # YOLO 모서리 교차 검증: 같은 카드를 가리키는 대비 검출 결과가 있으면
                # 네 변이 실제 경계선 위에 더 잘 놓인 쪽을 채택 (학습 안 된 카드에서 모서리가 틀리는 문제 방지)
                emap = edge_map(img)
                yq, _ = refine_quad(img, card_q)
                best_q, best_s = card_q, edge_score(emap, yq)
                for cq, kind in cands:
                    if _overlaps(cq, card_q):
                        rq, _ = refine_quad(img, cq)
                        sc = edge_score(emap, rq)
                        if sc > best_s * 1.2:
                            best_q, best_s = cq, sc
                            method = f"대비 기반 검출 – {kind} (YOLO 모서리가 부정확해 교체)"
                card_q = best_q
                cands = []                                 # YOLO가 카드를 찾았으면 후보 선택은 쓰지 않음
            if card_q is None:
                if cands:
                    pick = min(card_pick, len(cands)) - 1
                    card_q, kind = cands[pick]
                    method = f"대비 기반 검출 – {kind} (후보 {pick + 1}/{len(cands)})"
            if card_q is None:
                card_q = detect_card_opencv(img)
                method = "OpenCV 사각형 검출"
            if card_q is None and box_q is not None:
                card_q = box_q          # 최후 수단: 박스 (정면 촬영일 때만 정확)
                method = "YOLO 박스 (원근 보정 불가 · 정면 촬영 가정)"
            if card_q is not None and use_refine:
                card_q, refined = refine_quad(img, card_q)
                method += " + 변 직선 보정" if refined else " (직선 보정 실패 → 원래 모서리 사용)"
            if card_q is not None:
                # 카드는 직사각형 → 사진에 찍혀도 '직사각형을 원근으로 본 모양'이어야 함
                card_q, reshaped = make_card_shape(img, edge_map(img), card_q)
                if reshaped:
                    method += " + 카드 모양 강제(찌그러진 사각형 → 카드 비율 직사각형으로 재맞춤)"

            res = model_crack.predict(source=img, conf=crack_conf, retina_masks=True, verbose=False)[0]
            yolo_mask = raster_mask(res, img.shape, card_q)
            mask = refine_crack(img, yolo_mask, card_q) if use_fine else yolo_mask
            crack_px = int(mask.sum())

        l, r = st.columns(2)
        l.subheader("🖼️ 원본")
        l.image(image, use_container_width=True)
        r.subheader("✨ 결과 (빨강: 균열, 하늘색: 카드)")
        r.image(draw(img, mask, card_q), use_container_width=True)

        if card_q is None:
            st.error("카드를 찾지 못했습니다. 카드 네 모서리가 잘 보이도록, 배경과 대비가 되게 다시 찍어주세요.")
            st.stop()
        if crack_px == 0:
            st.info("탐지된 균열이 없습니다. 균열 민감도를 낮춰보세요.")
            st.stop()

        # 방법 1: 단순 면적비 (정면 촬영일 때만 정확)
        card_px = cv2.contourArea(card_q)
        area_ratio_mm2 = crack_px * CARD_AREA_MM2 / card_px

        # 방법 2: 호모그래피 원근 보정 (권장)
        H = card_homography(card_q)
        area_h_mm2 = metric_area_mm2(mask, H)

        st.success(f"🚨 측정 완료 (카드 검출: {method})")
        if mask.mean() > 0.10:
            st.warning("균열로 칠해진 영역이 사진의 10%를 넘습니다. 카펫·거친 마감재처럼 표면 무늬가 균열과 비슷하면 "
                       "모델이 구별하지 못해 면적이 크게 부풀려집니다. 콘크리트 면에서 다시 촬영해 주세요.")
        if len(cands) > 1:
            st.warning(f"카드처럼 보이는 물체가 {len(cands)}개 있습니다. 하늘색 테두리가 카드가 아니면 "
                       "'카드 후보 번호'를 바꿔 다시 측정하세요.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("균열 픽셀 수", f"{crack_px:,} px")
        m2.metric("카드 픽셀 수", f"{int(card_px):,} px")
        m3.metric("균열 면적 (원근 보정)", f"{area_h_mm2:,.1f} mm²", f"{area_h_mm2/100:.3f} cm²", delta_color="off")
        m4.metric("참고: 단순 면적비", f"{area_ratio_mm2:,.1f} mm²", delta_color="off")
        if use_fine:
            yolo_area = metric_area_mm2(yolo_mask, H)
            st.caption(f"참고: 정밀화 전 YOLO 마스크 기준 면적 {yolo_area:,.1f} mm² "
                       f"(정밀화 후 {area_h_mm2 / max(yolo_area, 1e-6) * 100:.0f}%)")

        asp = quad_aspect(card_q)
        if abs(asp - CARD_RATIO) > 0.12:
            st.warning(f"검출된 카드 가로세로비 {asp:.2f} (정상 1.59). 많이 비스듬하거나 검출이 부정확할 수 있습니다.")
        diff = abs(area_ratio_mm2 - area_h_mm2) / max(area_h_mm2, 1e-6) * 100
        st.caption(f"두 방식 차이 {diff:.1f}% — 차이가 크면 사진이 비스듬하다는 뜻이며, 원근 보정 값을 사용하세요.")

# 실행: cd ~/Desktop → streamlit run crack_app_card.py