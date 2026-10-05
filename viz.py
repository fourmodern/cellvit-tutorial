"""튜토리얼 공통 시각화 도구.

- overlay / side_by_side : 조직 사진 위에 세포 윤곽·반투명 채움을 겹친 정적 그림
- cell_gallery           : 클래스별 실제 세포 사진 모음
- interactive            : plotly 뷰어 (확대/이동, 마우스 올리면 세포 정보, 범례로 클래스 켜고 끄기)
- interactive_layers     : 같은 조직에 여러 분류 결과를 버튼으로 바꿔 보는 plotly 뷰어
- interactive_compare    : 여러 패널의 확대/이동이 연동되는 plotly 뷰어
- wsi_view               : 슬라이드 썸네일 위 전체 세포 분포 (plotly, WebGL)
- region_explorer        : 슬라이더로 위치/크기를 골라 원본 해상도 조직 + 세포를 보는 탐색기 (ipywidgets)

세포(cell)는 dict이며 최소한 "contour"((N, 2) x, y 좌표)를 가진다.
색과 이름은 color_of(cell) -> (R, G, B), label_of(cell) -> str 함수로 넘긴다.
"""
from __future__ import annotations

import base64
import io
import math
from typing import Callable, Iterable, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from PIL import Image

RGB = tuple


# ─────────────────────────────────────────────────────────────
# 정적 그림 (matplotlib)
# ─────────────────────────────────────────────────────────────
def _cnt(c, offset=(0, 0)):
    return (np.round(np.asarray(c["contour"], dtype=float) - np.asarray(offset))).astype(np.int32).reshape(-1, 1, 2)


def overlay(img, cells, color_of: Callable, mode: str = "both", alpha: float = 0.35, thickness: int = 2, offset=(0, 0)):
    """조직 사진 위에 세포를 그린다.

    mode: "contour"(윤곽선만) | "fill"(반투명 채움) | "both"(채움 + 윤곽선)
    offset: 세포 좌표에서 뺄 값 (예: 이미지가 큰 좌표계의 일부일 때 그 왼쪽 위 좌표)
    """
    out = img.copy()
    if mode in ("fill", "both"):
        fill = img.copy()
        for c in cells:
            cv2.fillPoly(fill, [_cnt(c, offset)], tuple(int(v) for v in color_of(c)))
        out = cv2.addWeighted(fill, alpha, img, 1 - alpha, 0)
    if mode in ("contour", "both"):
        for c in cells:
            cv2.drawContours(out, [_cnt(c, offset)], -1, tuple(int(v) for v in color_of(c)), thickness)
    return out


def legend(ax, items: Iterable, fontsize=9, **kw):
    """items: [(이름, (R, G, B)), ...]"""
    handles = [Patch(color=np.array(c) / 255, label=n) for n, c in items]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=fontsize, **kw)


def side_by_side(img, panels: Sequence, crop=None, size: float = 7, suptitle: str | None = None):
    """조직 사진 원본과 오버레이 결과들을 같은 배율로 나란히 보여준다.

    panels: [(제목, 오버레이 이미지, 범례 items 또는 None), ...]  — 첫 칸에는 항상 원본 조직 사진
    crop: (slice_y, slice_x) 로 일부만 확대
    """
    crop = crop or (slice(None), slice(None))
    n = len(panels) + 1
    fig, ax = plt.subplots(1, n, figsize=(size * n, size * 1.02))
    ax[0].imshow(img[crop])
    ax[0].set_title("조직 사진 (H&E)")
    for a, (title, im, items) in zip(ax[1:], panels):
        a.imshow(im[crop])
        a.set_title(title)
        if items:
            legend(a, items)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    if suptitle:
        fig.suptitle(suptitle, fontsize=15)
    plt.tight_layout()
    plt.show()


def cell_gallery(cells, label_of: Callable, color_of: Callable, order: Sequence[str] | None = None,
                 img=None, get_crop: Callable | None = None, n: int = 10, size: int = 72,
                 seed: int = 0, title: str | None = None, offset=(0, 0)):
    """클래스별로 세포 n개씩 실제 조직 사진을 잘라 보여준다 (가운데 세포 윤곽 표시).

    img: 세포 좌표계와 같은 이미지 (offset만큼 이동) — 또는
    get_crop(cx, cy, size) -> RGB: 큰 슬라이드에서 세포 중심 주변을 직접 읽는 함수
    """
    rng = np.random.default_rng(seed)
    by = {}
    for c in cells:
        by.setdefault(label_of(c), []).append(c)
    order = [o for o in (order or sorted(by)) if o in by]
    if not order:
        print("표시할 세포가 없습니다."); return
    fig, axes = plt.subplots(len(order), n, figsize=(n * 1.15, len(order) * 1.25), squeeze=False)
    half = size // 2
    if img is not None:
        pad = np.pad(img, ((half, half), (half, half), (0, 0)), constant_values=225)  # 이미지 밖은 회색
    for r, lab in enumerate(order):
        pool = by[lab]
        pick = rng.choice(len(pool), min(n, len(pool)), replace=False)
        for j in range(n):
            a = axes[r, j]
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
            if j >= len(pick):
                continue
            c = pool[pick[j]]
            cx, cy = np.asarray(c["centroid"], float)
            if get_crop is not None:
                crop = get_crop(int(cx) - half, int(cy) - half, size)
                ox, oy = int(cx) - half, int(cy) - half
            else:
                lx, ly = int(cx - offset[0]), int(cy - offset[1])
                crop = pad[ly:ly + size, lx:lx + size].copy()
                ox, oy = int(cx) - half, int(cy) - half
            crop = np.ascontiguousarray(crop)
            cv2.drawContours(crop, [_cnt(c, (ox, oy))], -1, tuple(int(v) for v in color_of(c)), 1)
            a.imshow(crop)
        axes[r, 0].set_ylabel(f"{lab}\n(n={len(pool)})", rotation=0, ha="right", va="center", fontsize=10)
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────
# 인터랙티브 (plotly)
# ─────────────────────────────────────────────────────────────
def _img_trace(img, max_side, x0=0, y0=0):
    import plotly.graph_objects as go
    h, w = img.shape[:2]
    s = max(1, math.ceil(max(h, w) / max_side))
    show = np.ascontiguousarray(img[::s, ::s])
    buf = io.BytesIO()
    Image.fromarray(show).save(buf, format="JPEG", quality=88)
    src = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    return go.Image(source=src, x0=x0, y0=y0, dx=s, dy=s, hoverinfo="skip")


def _cell_traces(cells, color_of, label_of, hover_of=None, order=None, name_prefix="", visible=True,
                 alpha=0.25, legendgroup_prefix="", showlegend=True):
    import plotly.graph_objects as go
    by = {}
    for c in cells:
        by.setdefault(label_of(c), []).append(c)
    order = [o for o in (order or sorted(by)) if o in by]
    traces = []
    for lab in order:
        cs = by[lab]
        r, g, b = (int(v) for v in color_of(cs[0]))
        xs, ys = [], []
        for c in cs:
            cnt = np.asarray(c["contour"], float)
            xs += cnt[:, 0].tolist() + [cnt[0, 0], None]
            ys += cnt[:, 1].tolist() + [cnt[0, 1], None]
        grp = f"{legendgroup_prefix}{lab}"
        traces.append(go.Scatter(
            x=xs, y=ys, mode="lines", fill="toself", fillcolor=f"rgba({r},{g},{b},{alpha})",
            line=dict(color=f"rgb({r},{g},{b})", width=1.5), name=f"{name_prefix}{lab} ({len(cs)})",
            legendgroup=grp, hoverinfo="skip", visible=visible, showlegend=showlegend))
        cen = np.array([c["centroid"] for c in cs], float)
        text = [hover_of(c) if hover_of else lab for c in cs]
        traces.append(go.Scatter(
            x=cen[:, 0], y=cen[:, 1], mode="markers", marker=dict(size=6, color=f"rgb({r},{g},{b})", opacity=0.01),
            text=text, hovertemplate="%{text}<br>x=%{x:.0f}, y=%{y:.0f}<extra></extra>",
            legendgroup=grp, showlegend=False, visible=visible))
    return traces


def _layout(fig, title, height, extent=None):
    """extent = (x0, y0, w, h): 축 범위를 이미지에 딱 맞춰 여백 없이"""
    fig.update_layout(
        title=dict(text=title, y=0.99, yanchor="top"), height=height, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(itemsizing="constant", font=dict(size=11)), dragmode="pan",
        hoverlabel=dict(bgcolor="white"), plot_bgcolor="white",
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1)
    if extent is not None:
        x0, y0, w, h = extent
        fig.update_xaxes(range=[x0, x0 + w], constrain="domain")
        fig.update_yaxes(range=[y0 + h, y0], constrain="domain")
    else:
        fig.update_yaxes(autorange="reversed")
    return fig


def interactive(img, cells, color_of, label_of, hover_of=None, order=None, title="", offset=(0, 0),
                max_side=2048, height=750, show=True):
    """조직 사진 + 세포 윤곽 plotly 뷰어.

    - 마우스 휠/드래그로 확대·이동, 더블클릭으로 원래 크기
    - 세포 위에 마우스를 올리면 hover_of(cell) 정보
    - 범례 항목 클릭: 그 클래스 숨기기/보이기, 더블클릭: 그 클래스만 보기
    offset: img의 왼쪽 위가 세포 좌표계에서 어디인지 (x0, y0)
    """
    import plotly.graph_objects as go
    fig = go.Figure([_img_trace(img, max_side, *offset)] + _cell_traces(cells, color_of, label_of, hover_of, order))
    _layout(fig, title, height, (offset[0], offset[1], img.shape[1], img.shape[0]))
    fig.update_layout(modebar_add=["drawrect", "eraseshape"])
    if not show:
        return fig
    fig.show(config={"scrollZoom": True})


def interactive_layers(img, layers, title="", offset=(0, 0), max_side=2048, height=750, show=True):
    """같은 조직 위에 여러 분류 결과를 버튼으로 바꿔 보기.

    layers: [(버튼 이름, cells, color_of, label_of, hover_of, order), ...]
    """
    import plotly.graph_objects as go
    traces, counts = [_img_trace(img, max_side, *offset)], []
    for i, (name, cells, color_of, label_of, hover_of, order) in enumerate(layers):
        t = _cell_traces(cells, color_of, label_of, hover_of, order, visible=(i == 0), legendgroup_prefix=f"{i}:")
        traces += t
        counts.append(len(t))
    fig = go.Figure(traces)
    buttons = []
    for i, (name, *_rest) in enumerate(layers):
        vis = [True]
        for j, n in enumerate(counts):
            vis += [j == i] * n
        buttons.append(dict(label=name, method="update", args=[{"visible": vis}]))
    vis_none = [True] + [False] * sum(counts)
    buttons.append(dict(label="조직 사진만", method="update", args=[{"visible": vis_none}]))
    _layout(fig, title, height, (offset[0], offset[1], img.shape[1], img.shape[0]))
    fig.update_layout(margin=dict(t=95),
                      updatemenus=[dict(type="buttons", direction="right", x=0, y=1.0, xanchor="left", yanchor="bottom",
                                        pad=dict(b=4), buttons=buttons)])
    if not show:
        return fig
    fig.show(config={"scrollZoom": True})


def interactive_compare(img, panels, title="", offset=(0, 0), max_side=1024, height=620, show=True):
    """여러 패널의 확대/이동이 함께 움직이는 비교 뷰어. 첫 패널은 조직 사진만.

    panels: [(패널 제목, cells, color_of, label_of, hover_of, order), ...]
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    n = len(panels) + 1
    fig = make_subplots(rows=1, cols=n, shared_xaxes="all", shared_yaxes="all", horizontal_spacing=0.01,
                        subplot_titles=["조직 사진 (H&E)"] + [p[0] for p in panels])
    fig.add_trace(_img_trace(img, max_side, *offset), row=1, col=1)
    for k, (name, cells, color_of, label_of, hover_of, order) in enumerate(panels, start=2):
        fig.add_trace(_img_trace(img, max_side, *offset), row=1, col=k)
        for t in _cell_traces(cells, color_of, label_of, hover_of, order, name_prefix=f"[{name}] ",
                              legendgroup_prefix=f"{k}:"):
            fig.add_trace(t, row=1, col=k)
    _layout(fig, title, height, (offset[0], offset[1], img.shape[1], img.shape[0]))
    for k in range(1, n + 1):
        fig.update_yaxes(scaleanchor=f"x{'' if k == 1 else k}", row=1, col=k)
    fig.update_layout(margin=dict(t=80))
    if not show:
        return fig
    fig.show(config={"scrollZoom": True})


def wsi_view(thumb, ds, xy, labels, color_map: dict, title="", order=None, max_points=200_000, height=700,
             marker_size=3, show=True):
    """슬라이드 썸네일 위에 모든 세포 중심을 클래스별 색 점으로 (WebGL).

    thumb: 썸네일 RGB, ds: 썸네일 1px = level-0 몇 px, xy: (N, 2) level-0 좌표, labels: (N,) 클래스 이름
    """
    import plotly.graph_objects as go
    xy = np.asarray(xy, float); labels = np.asarray(labels)
    if len(xy) > max_points:
        idx = np.random.default_rng(0).choice(len(xy), max_points, replace=False)
        xy, labels = xy[idx], labels[idx]
    fig = go.Figure([_img_trace(thumb, 4096) if ds == 1 else _scaled_thumb(thumb, ds)])
    for lab in [o for o in (order or sorted(set(labels))) if o in set(labels)]:
        m = labels == lab
        r, g, b = (int(v) for v in color_map[lab])
        fig.add_trace(go.Scattergl(x=xy[m, 0], y=xy[m, 1], mode="markers", name=f"{lab} ({m.sum():,})",
                                   marker=dict(size=marker_size, color=f"rgb({r},{g},{b})"),
                                   hovertemplate=f"{lab}<br>x=%{{x:.0f}}, y=%{{y:.0f}}<extra></extra>"))
    _layout(fig, title, height, (0, 0, thumb.shape[1] * ds, thumb.shape[0] * ds))
    if not show:
        return fig
    fig.show(config={"scrollZoom": True})


def _scaled_thumb(thumb, ds):
    import plotly.graph_objects as go
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(thumb)).save(buf, format="JPEG", quality=88)
    src = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    return go.Image(source=src, x0=0, y0=0, dx=ds, dy=ds, hoverinfo="skip")


# ─────────────────────────────────────────────────────────────
# 슬라이드 탐색기 (ipywidgets)
# ─────────────────────────────────────────────────────────────
def region_explorer(read_rgb: Callable, cells_xy, cells, color_of, label_of, legend_items,
                    thumb, ds, x_range, y_range, init=None, sizes=(256, 512, 1024, 2048), mode="both"):
    """슬라이더로 위치·크기를 골라 원본 해상도 조직 사진 + 세포를 본다 (Colab/Jupyter).

    read_rgb(x, y, w, h) -> RGB : 원본 해상도 영역 읽기 (색 정규화 포함 가능)
    cells_xy: (N, 2) 세포 중심 level-0 좌표,  cells: 같은 순서의 세포 dict (contour는 level-0 좌표)
    thumb, ds: 위치 표시용 썸네일과 축소 배율
    """
    import ipywidgets as W
    from IPython.display import display

    cells_xy = np.asarray(cells_xy, float)
    x0, y0 = init or ((x_range[0] + x_range[1]) // 2, (y_range[0] + y_range[1]) // 2)
    sx = W.IntSlider(value=x0, min=x_range[0], max=x_range[1], step=64, description="x 중심", continuous_update=False,
                     layout=W.Layout(width="45%"))
    sy = W.IntSlider(value=y0, min=y_range[0], max=y_range[1], step=64, description="y 중심", continuous_update=False,
                     layout=W.Layout(width="45%"))
    ss = W.Dropdown(options=list(sizes), value=sizes[min(2, len(sizes) - 1)], description="영역(px)")
    sm = W.ToggleButtons(options=[("채움+윤곽", "both"), ("윤곽만", "contour"), ("조직만", "none")], value=mode,
                         description="표시")
    out = W.Output()

    def draw(size, x, y, mode_):
        img = read_rgb(x, y, size, size)
        m = ((cells_xy[:, 0] >= x) & (cells_xy[:, 0] < x + size) & (cells_xy[:, 1] >= y) & (cells_xy[:, 1] < y + size))
        sel = [cells[i] for i in np.flatnonzero(m)]
        fig, ax = plt.subplots(1, 3, figsize=(21, 7.4), gridspec_kw={"width_ratios": [0.9, 1, 1]})
        ax[0].imshow(thumb)
        ax[0].add_patch(plt.Rectangle((x / ds, y / ds), max(size / ds, 3), max(size / ds, 3), fill=False, ec="lime", lw=2))
        ax[0].set_title("위치")
        ax[1].imshow(img); ax[1].set_title(f"조직 사진 {size}×{size}px")
        ax[2].imshow(img if mode_ == "none" else overlay(img, sel, color_of, mode=mode_, offset=(x, y),
                                                        thickness=1 if size >= 1024 else 2))
        ax[2].set_title(f"세포 {len(sel)}개")
        legend(ax[2], legend_items)
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
        plt.tight_layout(); plt.show()

    def render(*_):
        size = ss.value
        with out:
            out.clear_output()
            draw(size, sx.value - size // 2, sy.value - size // 2, sm.value)

    for w in (sx, sy, ss, sm):
        w.observe(render, names="value")
    display(W.VBox([W.HBox([sx, sy]), W.HBox([ss, sm]), out]))
    # 첫 화면은 일반 출력으로 그린다 (Output 위젯 안에서 그리면 일부 실행 환경에서 멈춤).
    # 슬라이더를 움직이면 위젯 영역에 새 그림이 나타난다.
    with out:
        print("슬라이더를 움직이면 여기에 새 영역이 그려집니다. 아래는 처음 위치입니다.")
    size = ss.value
    draw(size, sx.value - size // 2, sy.value - size // 2, sm.value)


# ─────────────────────────────────────────────────────────────
# 분포 · 정확도 그래프
# ─────────────────────────────────────────────────────────────
def _c(col):
    return np.array(col) / 255


def type_summary(df, type_col: str, color_map: dict, order: Sequence[str] | None = None,
                 value_cols: dict | None = None, title: str | None = None, log_count: bool = False):
    """세포 타입별 분포 대시보드.

    - 타입별 세포 수(막대, 숫자 표시)와 비율(도넛)
    - value_cols의 각 수치(예: 모델 확신도, 핵 면적)를 타입별 바이올린 + 중앙값으로
    df: 세포 하나당 한 행. color_map: {타입: (R, G, B)}. value_cols: {열 이름: 축 제목}
    """
    value_cols = value_cols or {}
    counts = df[type_col].value_counts()
    order = [o for o in (order or counts.index.tolist()) if o in counts.index]
    n_val = len(value_cols)
    fig = plt.figure(figsize=(6 + 5 + 6 * n_val, max(4.5, 0.38 * len(order) + 1.5)))
    gs = fig.add_gridspec(1, 2 + n_val, width_ratios=[6, 4.2] + [6] * n_val)

    ax = fig.add_subplot(gs[0])
    vals = [counts[o] for o in order]
    ax.barh(order, vals, color=[_c(color_map[o]) for o in order], ec="k", lw=0.5)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:,}", va="center", fontsize=9)
    ax.invert_yaxis(); ax.set_xlabel("세포 수"); ax.set_title(f"타입별 세포 수 (총 {len(df):,})")
    if log_count:
        ax.set_xscale("log")
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(gs[1])
    frac = np.array(vals) / sum(vals)
    wedges, _ = ax.pie(vals, colors=[_c(color_map[o]) for o in order], startangle=90, counterclock=False,
                       wedgeprops=dict(width=0.42, edgecolor="w"))
    for w, f, o in zip(wedges, frac, order):
        if f >= 0.04:
            ang = np.deg2rad((w.theta1 + w.theta2) / 2)
            ax.text(0.79 * np.cos(ang), 0.79 * np.sin(ang), f"{f:.0%}", ha="center", va="center", fontsize=8)
    ax.set_title("비율")

    for k, (col, label) in enumerate(value_cols.items()):
        ax = fig.add_subplot(gs[2 + k])
        data = [df.loc[df[type_col] == o, col].dropna().values for o in order]
        pos = np.arange(len(order))
        ok = [i for i, d in enumerate(data) if len(d) > 1]
        if ok:
            parts = ax.violinplot([data[i] for i in ok], positions=pos[ok], vert=False, widths=0.8, showextrema=False)
            for i, b in zip(ok, parts["bodies"]):
                b.set_facecolor(_c(color_map[order[i]])); b.set_edgecolor("k"); b.set_alpha(0.75)
        med = [np.median(d) if len(d) else np.nan for d in data]
        ax.scatter(med, pos, color="k", s=14, zorder=3, label="중앙값")
        ax.set_yticks(pos); ax.set_yticklabels(order, fontsize=9)
        ax.invert_yaxis(); ax.set_xlabel(label); ax.set_title(f"타입별 {label}")
        ax.grid(axis="x", alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def classification_metrics(y_true, y_pred, labels: dict, color_map: dict, title: str | None = None):
    """정답이 있을 때: 클래스별 precision / recall / F1 막대 + 정규화 혼동행렬.

    labels: {클래스 id: 이름}, color_map: {클래스 id: (R, G, B)}
    """
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
    ids = list(labels)
    names = [labels[i] for i in ids]
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=ids, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, labels=ids, average="macro", zero_division=0)

    fig, ax = plt.subplots(1, 2, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})
    x = np.arange(len(ids)); w = 0.26
    for k, (vals, lab, hatch) in enumerate([(p, "Precision (정밀도)", ""), (r, "Recall (재현율)", "//"), (f, "F1", "..")]):
        bars = ax[0].bar(x + (k - 1) * w, vals, w, label=lab, color=[_c(color_map[i]) for i in ids],
                         edgecolor="k", hatch=hatch, alpha=0.55 + 0.2 * k)
        for b, v in zip(bars, vals):
            ax[0].text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7.5)
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"{n}\n(n={k})" for n, k in zip(names, s)])
    ax[0].set_ylim(0, 1.1); ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=9)
    ax[0].axhline(mf1, color="k", ls="--", lw=1); ax[0].text(-0.45, mf1 + 0.015, f"macro-F1 {mf1:.2f}", ha="left", fontsize=9, bbox=dict(fc="white", ec="none", alpha=0.8))
    ax[0].set_title(f"클래스별 성능 (정확도 {acc:.1%}, macro-F1 {mf1:.2f})")
    ax[0].spines[["top", "right"]].set_visible(False)

    cm = confusion_matrix(y_true, y_pred, labels=ids)
    pct = cm / cm.sum(1, keepdims=True).clip(min=1)
    ax[1].imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(ids)):
        for j in range(len(ids)):
            ax[1].text(j, i, f"{pct[i, j]:.0%}\n({cm[i, j]})", ha="center", va="center", fontsize=9,
                       color="w" if pct[i, j] > 0.6 else "k")
    ax[1].set_xticks(range(len(ids))); ax[1].set_xticklabels(names, rotation=25, ha="right")
    ax[1].set_yticks(range(len(ids))); ax[1].set_yticklabels(names)
    ax[1].set_xlabel("예측"); ax[1].set_ylabel("정답"); ax[1].set_title("혼동행렬 (행 = 정답 기준 %)")
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def confidence_vs_accuracy(conf, correct, bins: int = 8, title: str | None = None):
    """모델 확신도 구간별 실제 정답률 (신뢰도 곡선) + 확신도 분포."""
    conf = np.asarray(conf, float); correct = np.asarray(correct, bool)
    edges = np.quantile(conf, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = 0, 1.0001
    idx = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    mid = np.array([conf[idx == b].mean() if (idx == b).any() else np.nan for b in range(bins)])
    acc = np.array([correct[idx == b].mean() if (idx == b).any() else np.nan for b in range(bins)])
    n = np.array([(idx == b).sum() for b in range(bins)])

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="완벽한 보정")
    ax[0].plot(mid, acc, "o-", color="#c33", label="실제 정답률")
    for x_, y_, k in zip(mid, acc, n):
        ax[0].text(x_, y_ + 0.03, f"n={k}", ha="center", fontsize=8)
    ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1.08); ax[0].set_xlabel("모델 확신도 (구간 평균)"); ax[0].set_ylabel("정답률")
    ax[0].set_title("확신도가 높을수록 더 정확한가?"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].hist([conf[correct], conf[~correct]], bins=20, stacked=True, color=["#3a7", "#c55"], label=["정답", "오답"])
    ax[1].set_xlabel("모델 확신도"); ax[1].set_ylabel("세포 수"); ax[1].set_title("확신도 분포"); ax[1].legend()
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def agreement_by_type(df, type_col: str, rate_cols: dict, color_map: dict, order: Sequence[str] | None = None,
                      title: str | None = None):
    """정답이 없을 때: 타입별로 다른 모델과의 일치율(검출·분류)을 막대로.

    df: 세포 하나당 한 행, rate_cols: {0/1 열 이름: 표시 이름}
    """
    g = df.groupby(type_col)
    order = [o for o in (order or g.size().sort_values(ascending=False).index.tolist()) if o in g.groups]
    k = len(rate_cols)
    fig, ax = plt.subplots(1, k, figsize=(7 * k, max(4, 0.4 * len(order) + 1.5)), squeeze=False)
    for a, (col, label) in zip(ax[0], rate_cols.items()):
        rate = g[col].mean().reindex(order); n = g[col].count().reindex(order)
        a.barh(order, rate, color=[_c(color_map[o]) for o in order], ec="k", lw=0.5)
        for i, (v, m) in enumerate(zip(rate, n)):
            if not np.isnan(v):
                a.text(v + 0.01, i, f"{v:.0%} (n={m})", va="center", fontsize=8.5)
        overall = df[col].mean()
        a.axvline(overall, color="k", ls="--", lw=1); a.text(overall, -0.7, f"전체 {overall:.0%}", ha="center", fontsize=9)
        a.set_xlim(0, 1.18); a.invert_yaxis(); a.set_title(label); a.set_xlabel("비율")
        a.spines[["top", "right"]].set_visible(False)
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────
# 패치(타일) 단위 결과 — foundation model 노트북용
# ─────────────────────────────────────────────────────────────
def image_grid(images, titles=None, ncols=8, size=1.6, border_colors=None, suptitle=None, row_labels=None):
    """이미지 여러 장을 격자로. border_colors: 각 이미지 테두리 색 (R, G, B) 또는 None"""
    n = len(images)
    nrows = math.ceil(n / ncols)
    lines = max((str(tt).count("\n") + 1 for tt in titles), default=0) if titles is not None else 0
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * size, nrows * (size + 0.25 + 0.17 * lines)), squeeze=False)
    for k, a in enumerate(axes.ravel()):
        a.set_xticks([]); a.set_yticks([])
        if k >= n:
            a.axis("off"); continue
        a.imshow(images[k])
        if titles is not None:
            a.set_title(titles[k], fontsize=8)
        col = border_colors[k] if border_colors is not None else None
        for s in a.spines.values():
            s.set_visible(col is not None)
            if col is not None:
                s.set_edgecolor(np.array(col) / 255); s.set_linewidth(3)
    if row_labels:
        for r, lab in enumerate(row_labels):
            axes[r, 0].set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=10)
    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout()
    plt.show()


def _tile_overlay(thumb, ds, xy, tile, rgb, alpha):
    """썸네일 위에 타일별 색을 반투명하게 칠한 이미지"""
    over = thumb.copy()
    s = max(1, int(round(tile / ds)))
    for (x, y), c in zip(np.asarray(xy), rgb):
        x0, y0 = int(x / ds), int(y / ds)
        over[y0:y0 + s, x0:x0 + s] = c
    return (thumb * (1 - alpha) + over * alpha).astype(np.uint8)


def tile_map(thumb, ds, xy, tile, labels=None, color_map=None, values=None, cmap="magma", vmin=None, vmax=None,
             alpha=0.55, title="", ax=None, legend_order=None, colorbar_label=None):
    """슬라이드 썸네일 위에 타일 결과를 칠한 정적 지도.

    labels + color_map: 범주형 (예: 조직 타입) / values + cmap: 연속형 (예: 확률, 유사도)
    xy: 타일 왼쪽 위 level-0 좌표, tile: 타일 한 변 level-0 px
    """
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(14, 14 * thumb.shape[0] / thumb.shape[1] + 0.6))
    if labels is not None:
        rgb = np.array([color_map[l] for l in labels], float)
        ax.imshow(_tile_overlay(thumb, ds, xy, tile, rgb, alpha))
        present = [l for l in (legend_order or sorted(set(labels))) if l in set(labels)]
        legend(ax, [(l, color_map[l]) for l in present])
    else:
        values = np.asarray(values, float)
        lo = np.nanmin(values) if vmin is None else vmin
        hi = np.nanmax(values) if vmax is None else vmax
        cm = plt.get_cmap(cmap)
        rgb = cm(np.clip((values - lo) / (hi - lo + 1e-9), 0, 1))[:, :3] * 255
        ax.imshow(_tile_overlay(thumb, ds, xy, tile, rgb, alpha))
        sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(lo, hi))
        plt.colorbar(sm, ax=ax, fraction=0.025, label=colorbar_label)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    if own:
        plt.tight_layout(); plt.show()


def tile_map_interactive(thumb, ds, xy, tile, labels, color_map, hover=None, title="", order=None, height=650, alpha=0.5):
    """타일 지도 plotly 버전: 확대/이동, 타일에 마우스를 올리면 hover 정보, 범례로 타입 켜고 끄기."""
    import plotly.graph_objects as go
    xy = np.asarray(xy, float); labels = np.asarray(labels)
    fig = go.Figure([_scaled_thumb(thumb, ds)])
    hover = np.asarray(hover) if hover is not None else labels
    for lab in [o for o in (order or sorted(set(labels))) if o in set(labels)]:
        m = labels == lab
        r, g, b = (int(v) for v in color_map[lab])
        xs, ys = [], []
        for x, y in xy[m]:
            xs += [x, x + tile, x + tile, x, x, None]; ys += [y, y, y + tile, y + tile, y, None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself", fillcolor=f"rgba({r},{g},{b},{alpha})",
                                 line=dict(width=0), name=f"{lab} ({m.sum()})", legendgroup=lab, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xy[m, 0] + tile / 2, y=xy[m, 1] + tile / 2, mode="markers",
                                 marker=dict(size=4, opacity=0.01), text=hover[m], legendgroup=lab, showlegend=False,
                                 hovertemplate="%{text}<extra></extra>"))
    _layout(fig, title, height, (0, 0, thumb.shape[1] * ds, thumb.shape[0] * ds))
    fig.show(config={"scrollZoom": True})
