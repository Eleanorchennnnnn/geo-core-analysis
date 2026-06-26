import sys
from colorama import init
init()
from flask import Flask, render_template, request, redirect, url_for, send_file, session, jsonify, send_from_directory
from flask_session import Session
import os
import cv2
import numpy as np
import math
import json
from datetime import datetime
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import uuid

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600

Session(app)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)


class CoreImageAnalyzer:
    def __init__(self, image_path, max_size=1200, core_width_cm=10.0):
        self.orig_image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if self.orig_image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        h, w = self.orig_image.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            self.image = cv2.resize(self.orig_image, (0, 0), fx=scale, fy=scale)
        else:
            self.image = self.orig_image.copy()
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.holes = []
        self.cracks = []
        self.granularities = []
        self.h, self.w = self.gray.shape
        self.pixel_to_cm = core_width_cm / max(self.w * 0.8, 1)
        self.cm_to_pixel = (self.w * 0.8) / max(core_width_cm, 0.1)

    # ==================== 孔洞检测（保持原有算法不变）====================
    def detect_holes(self, threshold=79, min_area=50):
        self.holes = []
        h, w = self.h, self.w
        gray = self.gray.flatten()
        binary = gray < threshold
        denoised = binary.copy()
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                idx = y * w + x
                if binary[idx]:
                    cnt = 0
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            if binary[(y + dy) * w + (x + dx)]:
                                cnt += 1
                    if cnt < 2:
                        denoised[idx] = False
        labels = np.full(w * h, -1, dtype=np.int32)
        regions = []
        for y in range(h):
            for x in range(w):
                idx = y * w + x
                if denoised[idx] and labels[idx] == -1:
                    region = []
                    queue = [idx]
                    labels[idx] = len(regions)
                    region.append(idx)
                    while queue:
                        cur = queue.pop(0)
                        cx = cur % w
                        cy = cur // w
                        dxs = [-1, 1, 0, 0]
                        dys = [0, 0, -1, 1]
                        for d in range(4):
                            nx = cx + dxs[d]
                            ny = cy + dys[d]
                            if 0 <= nx < w and 0 <= ny < h:
                                nidx = ny * w + nx
                                if denoised[nidx] and labels[nidx] == -1:
                                    labels[nidx] = len(regions)
                                    queue.append(nidx)
                                    region.append(nidx)
                    regions.append(region)
        for region in regions:
            area = len(region)
            if area < min_area:
                continue
            contour = self._extract_contour(region, w, h)
            if len(contour) < 3:
                continue
            cnt_array = np.array(contour, dtype=np.int32).reshape((-1, 1, 2))
            M = cv2.moments(cnt_array)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            diameter = 2 * math.sqrt(area / math.pi)
            self.holes.append({
                'id': len(self.holes) + 1,
                'area': int(area),
                'area_cm2': float(area * (self.pixel_to_cm ** 2)),
                'diameter': float(diameter),
                'diameter_cm': float(diameter * self.pixel_to_cm),
                'position': (int(cx), int(cy)),
                'contour': cnt_array
            })
        return self.holes

    # ==================== 裂缝检测（按Java CrackAnalyzer精确翻译）====================
    def detect_cracks(self, threshold=97, min_points=10):
        """
        按Java CrackAnalyzer算法：
        1. 高斯平滑（3x3核 [1,2,1,2,4,2,1,2,1]/16）
        2. 全局阈值：暗区检测（smoothed < threshold）
        3. 去除边缘margin=5
        4. 去噪（邻居<2的去掉）
        5. 膨胀2次
        6. Zhang-Suen骨架化
        7. 8连通BFS标记区域
        8. 筛选：区域点数>=10，且（长宽比>2.5 或 长度>40）
        """
        self.cracks = []
        h, w = self.h, self.w
        margin = 5

        # 1. 高斯平滑（与Java完全一致：3x3核 [1,2,1,2,4,2,1,2,1]/16）
        gray = self.gray.flatten().astype(np.float64)
        smoothed = np.zeros_like(gray)
        kernel = np.array([1, 2, 1, 2, 4, 2, 1, 2, 1], dtype=np.float64)
        ksum = 16.0
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                val = 0.0
                for ky in range(-1, 2):
                    for kx in range(-1, 2):
                        ki = (ky + 1) * 3 + (kx + 1)
                        val += gray[(y + ky) * w + (x + kx)] * kernel[ki]
                smoothed[y * w + x] = val / ksum

        # 2. 暗区检测：裂缝是暗线
        dark = smoothed < threshold

        # 3. 去除边缘
        if margin > 0:
            for y in range(h):
                for x in range(w):
                    if x < margin or x >= w - margin or y < margin or y >= h - margin:
                        dark[y * w + x] = False

        # 4. 去噪（邻居<2的去掉）—— 注意：包含中心点本身，所以cnt<2意味着只有自身
        filtered = dark.copy()
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                idx = y * w + x
                if dark[idx]:
                    cnt = 0
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            if dark[(y + dy) * w + (x + dx)]:
                                cnt += 1
                    if cnt < 2:
                        filtered[idx] = False

        # 5. 膨胀2次
        dilated = self._dilate(filtered, w, h)
        dilated = self._dilate(dilated, w, h)

        # 6. Zhang-Suen骨架化
        skeleton = self._zhang_suen_thinning(dilated, w, h)

        # 7. 8连通BFS标记区域
        labels = np.full(w * h, -1, dtype=np.int32)
        regions = []
        for y in range(h):
            for x in range(w):
                idx = y * w + x
                if skeleton[idx] and labels[idx] == -1:
                    region = []
                    queue = [idx]
                    labels[idx] = len(regions)
                    region.append(idx)
                    while queue:
                        cur = queue.pop(0)
                        cx = cur % w
                        cy = cur // w
                        for dy in range(-1, 2):
                            for dx in range(-1, 2):
                                if dx == 0 and dy == 0:
                                    continue
                                nx = cx + dx
                                ny = cy + dy
                                if 0 <= nx < w and 0 <= ny < h:
                                    nidx = ny * w + nx
                                    if skeleton[nidx] and labels[nidx] == -1:
                                        labels[nidx] = len(regions)
                                        queue.append(nidx)
                                        region.append(nidx)
                    regions.append(region)

        # 8. 筛选裂缝
        for region in regions:
            if len(region) < min_points:
                continue

            min_x, max_x = w, 0
            min_y, max_y = h, 0
            for idx in region:
                px = idx % w
                py = idx // w
                if px < min_x:
                    min_x = px
                if px > max_x:
                    max_x = px
                if py < min_y:
                    min_y = py
                if py > max_y:
                    max_y = py

            bbox_w = max_x - min_x + 1
            bbox_h = max_y - min_y + 1
            length = math.sqrt(bbox_w * bbox_w + bbox_h * bbox_h)
            width = min(bbox_w, bbox_h)
            aspect_ratio = length / max(width, 1)

            # 放宽条件：细长或足够长
            is_crack = aspect_ratio > 2.5 or length > 40
            if not is_crack:
                continue

            # 保存骨架像素点用于绘制
            skeleton_pixels = []
            for idx in region:
                skeleton_pixels.append((idx % w, idx // w))

            self.cracks.append({
                'id': len(self.cracks) + 1,
                'length': float(length),
                'length_cm': float(length * self.pixel_to_cm),
                'width': float(width),
                'width_cm': float(width * self.pixel_to_cm),
                'aspect_ratio': float(aspect_ratio),
                'fractal_dim': 1.0,
                'type': "裂缝",
                'position': (int(min_x), int(max(0, min_y - 5))),
                'skeleton_pixels': skeleton_pixels,
                'contour': None
            })

        return self.cracks

    # ==================== 粒度检测====================
    def detect_granularities(self, threshold=138, min_area=15):
        self.granularities = []
        h, w = self.h, self.w
        gray = self.gray.flatten()

        # 1. 计算全局平均灰度
        global_avg = int(np.mean(gray))

        # 2. 自适应阈值：21x21局部窗口 + 全局平均
        window_size = 21
        offset = threshold - 127
        binary = np.zeros(w * h, dtype=bool)

        # 使用 cv2.boxFilter 计算局部均值（与21x21窗口一致，且稳定兼容）
        local_avg_img = cv2.boxFilter(self.gray, -1, (window_size, window_size),
                                      normalize=True, borderType=cv2.BORDER_REFLECT)
        local_avg_flat = local_avg_img.flatten().astype(np.int32)

        for idx in range(w * h):
            local_avg = local_avg_flat[idx]
            # 比局部平均亮，且比全局平均亮
            binary[idx] = (gray[idx] > local_avg + offset and
                           gray[idx] > global_avg + offset)

        # 3. 去噪（邻居<2的去掉）
        denoised = binary.copy()
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                idx = y * w + x
                if binary[idx]:
                    cnt = 0
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            if binary[(y + dy) * w + (x + dx)]:
                                cnt += 1
                    if cnt < 2:
                        denoised[idx] = False

        # 4. 膨胀2次
        dilated = self._dilate(denoised, w, h)
        dilated = self._dilate(dilated, w, h)

        # 5. 8连通BFS标记区域
        labels = np.full(w * h, -1, dtype=np.int32)
        regions = []
        for y in range(h):
            for x in range(w):
                idx = y * w + x
                if dilated[idx] and labels[idx] == -1:
                    region = []
                    queue = [idx]
                    labels[idx] = len(regions)
                    region.append(idx)
                    while queue:
                        cur = queue.pop(0)
                        cx = cur % w
                        cy = cur // w
                        for dy in range(-1, 2):
                            for dx in range(-1, 2):
                                if dx == 0 and dy == 0:
                                    continue
                                nx = cx + dx
                                ny = cy + dy
                                if 0 <= nx < w and 0 <= ny < h:
                                    nidx = ny * w + nx
                                    if dilated[nidx] and labels[nidx] == -1:
                                        labels[nidx] = len(regions)
                                        queue.append(nidx)
                                        region.append(nidx)
                    regions.append(region)

        # 6. 计算粒径并筛选
        for region in regions:
            area = len(region)
            if area < min_area:
                continue

            diameter = 2 * math.sqrt(area / math.pi)

            # 提取轮廓
            contour = self._extract_contour(region, w, h)
            if len(contour) < 3:
                continue
            cnt_array = np.array(contour, dtype=np.int32).reshape((-1, 1, 2))

            # 计算圆度
            perimeter = cv2.arcLength(cnt_array, True)
            circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

            # 找最小位置用于标注
            min_x, min_y = w, h
            for idx in region:
                px = idx % w
                py = idx // w
                if px < min_x:
                    min_x = px
                if py < min_y:
                    min_y = py

            self.granularities.append({
                'id': len(self.granularities) + 1,
                'area': float(area),
                'area_cm2': float(area * (self.pixel_to_cm ** 2)),
                'size': float(diameter),
                'size_cm': float(diameter * self.pixel_to_cm),
                'circularity': float(circularity),
                'position': (int(min_x), int(max(0, min_y - 3))),
                'contour': cnt_array
            })

        return self.granularities

    # ==================== 膨胀操作====================
    def _dilate(self, src, w, h):
        dst = src.copy()
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                idx = y * w + x
                if not src[idx]:
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            if src[(y + dy) * w + (x + dx)]:
                                dst[idx] = True
                                break
                        if dst[idx]:
                            break
        return dst

    # ==================== Zhang-Suen骨架化====================
    def _zhang_suen_thinning(self, src, w, h):
        dst = src.copy()
        changed = True
        while changed:
            changed = False
            to_remove = []
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    idx = y * w + x
                    if not dst[idx]:
                        continue
                    p = [0] * 8
                    p[0] = 1 if dst[(y - 1) * w + (x - 1)] else 0
                    p[1] = 1 if dst[(y - 1) * w + x] else 0
                    p[2] = 1 if dst[(y - 1) * w + (x + 1)] else 0
                    p[3] = 1 if dst[y * w + (x + 1)] else 0
                    p[4] = 1 if dst[(y + 1) * w + (x + 1)] else 0
                    p[5] = 1 if dst[(y + 1) * w + x] else 0
                    p[6] = 1 if dst[(y + 1) * w + (x - 1)] else 0
                    p[7] = 1 if dst[y * w + (x - 1)] else 0
                    A = 0
                    for i in range(8):
                        if p[i] == 0 and p[(i + 1) % 8] == 1:
                            A += 1
                    B = sum(p)
                    if B >= 2 and B <= 6 and A == 1 and p[0] * p[2] * p[4] == 0 and p[2] * p[4] * p[6] == 0:
                        to_remove.append(idx)
            for idx in to_remove:
                dst[idx] = False
                changed = True
            to_remove = []
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    idx = y * w + x
                    if not dst[idx]:
                        continue
                    p = [0] * 8
                    p[0] = 1 if dst[(y - 1) * w + (x - 1)] else 0
                    p[1] = 1 if dst[(y - 1) * w + x] else 0
                    p[2] = 1 if dst[(y - 1) * w + (x + 1)] else 0
                    p[3] = 1 if dst[y * w + (x + 1)] else 0
                    p[4] = 1 if dst[(y + 1) * w + (x + 1)] else 0
                    p[5] = 1 if dst[(y + 1) * w + x] else 0
                    p[6] = 1 if dst[(y + 1) * w + (x - 1)] else 0
                    p[7] = 1 if dst[y * w + (x - 1)] else 0
                    A = 0
                    for i in range(8):
                        if p[i] == 0 and p[(i + 1) % 8] == 1:
                            A += 1
                    B = sum(p)
                    if B >= 2 and B <= 6 and A == 1 and p[0] * p[2] * p[6] == 0 and p[0] * p[4] * p[6] == 0:
                        to_remove.append(idx)
            for idx in to_remove:
                dst[idx] = False
                changed = True
        return dst

    # ==================== 轮廓提取====================
    def _extract_contour(self, region, w, h):
        in_region = np.zeros(w * h, dtype=bool)
        for idx in region:
            in_region[idx] = True
        start_idx = -1
        min_x, min_y = w, h
        for idx in region:
            px = idx % w
            py = idx // w
            if py < min_y or (py == min_y and px < min_x):
                min_x = px
                min_y = py
                start_idx = idx
        if start_idx == -1:
            return []
        contour = []
        dx = [1, 1, 0, -1, -1, -1, 0, 1]
        dy = [0, 1, 1, 1, 0, -1, -1, -1]
        cx, cy = min_x, min_y
        start_dir = 6
        while True:
            contour.append([cx, cy])
            found = False
            for d in range(8):
                dir_idx = (start_dir + d) % 8
                nx = cx + dx[dir_idx]
                ny = cy + dy[dir_idx]
                if 0 <= nx < w and 0 <= ny < h and in_region[ny * w + nx]:
                    cx, cy = nx, ny
                    start_dir = (dir_idx + 5) % 8
                    found = True
                    break
            if not found:
                break
            if cx == min_x and cy == min_y:
                break
        return contour

    # ==================== 绘制结果 ====================
    def draw_analysis_results(self, filter_type='all'):
        result_img = self.image.copy()
        if filter_type in ['all', 'holes']:
            for hole in self.holes:
                cv2.drawContours(result_img, [hole['contour']], -1, (0, 0, 255), 2)
                cv2.putText(result_img, f"H{hole['id']}",
                           (hole['position'][0] + 10, hole['position'][1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if filter_type in ['all', 'grains']:
            for grain in self.granularities:
                cv2.drawContours(result_img, [grain['contour']], -1, (0, 255, 0), 2)
                cv2.putText(result_img, f"G{grain['id']}",
                           (grain['position'][0] + 5, grain['position'][1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if filter_type in ['all', 'cracks']:
            for crack in self.cracks:
                # 直接画骨架像素点（蓝色）
                if crack.get('skeleton_pixels'):
                    for (px, py) in crack['skeleton_pixels']:
                        cv2.circle(result_img, (px, py), 1, (255, 0, 0), -1)
                elif crack.get('contour') is not None:
                    cv2.drawContours(result_img, [crack['contour']], -1, (255, 0, 0), 2)
                cv2.putText(result_img, f"C{crack['id']}",
                        (int(crack['position'][0]), int(crack['position'][1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        return result_img


# ==================== Flask 路由 ====================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "没有上传文件", 400
    file = request.files['image']
    if file.filename == '':
        return "未选择文件", 400
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    session['image_path'] = filepath
    session['image_filename'] = filename
    return redirect(url_for('analyze'))


@app.route('/analyze')
def analyze():
    if 'image_path' not in session:
        return redirect(url_for('index'))
    return render_template('analyze.html', image_path=session['image_path'])


@app.route('/process', methods=['POST'])
def process():
    if 'image_path' not in session:
        return jsonify({'error': '请先上传图像'}), 400

    filepath = session['image_path']

    # 获取分析类型参数
    data = request.get_json(silent=True) or {}
    analysis_type = data.get('analysis_type', 'all')
    valid_types = {'all', 'holes', 'cracks', 'grains'}
    if analysis_type not in valid_types:
        analysis_type = 'all'

    try:
        analysis_params = {
            'hole_threshold': 79,
            'hole_min_area': 50,
            'crack_threshold': 97,
            'crack_min_points': 10,
            'grain_threshold': 138,
            'grain_min_area': 15,
            'analysis_type': analysis_type
        }

        analyzer = CoreImageAnalyzer(filepath, core_width_cm=10.0)

        # 根据选择的分析类型执行对应的检测
        holes = []
        cracks = []
        grains = []

        if analysis_type in ('all', 'holes'):
            holes = analyzer.detect_holes(
                threshold=analysis_params['hole_threshold'],
                min_area=analysis_params['hole_min_area']
            )
        if analysis_type in ('all', 'cracks'):
            cracks = analyzer.detect_cracks(
                threshold=analysis_params['crack_threshold'],
                min_points=analysis_params['crack_min_points']
            )
        if analysis_type in ('all', 'grains'):
            grains = analyzer.detect_granularities(
                threshold=analysis_params['grain_threshold'],
                min_area=analysis_params['grain_min_area']
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = str(uuid.uuid4())[:8]

        def save_result_image(img, suffix):
            filename = f'result_{suffix}_{timestamp}_{session_id}.jpg'
            path = os.path.join(app.config['RESULT_FOLDER'], filename)
            success, encoded = cv2.imencode('.jpg', img)
            if success:
                with open(path, 'wb') as f:
                    f.write(encoded.tobytes())
            return path, filename

        # 根据分析类型生成对应的结果图
        if analysis_type == 'all':
            result_img = analyzer.draw_analysis_results(filter_type='all')
            result_path, result_filename = save_result_image(result_img, 'all')
            holes_img = analyzer.draw_analysis_results(filter_type='holes')
            holes_path, holes_filename = save_result_image(holes_img, 'holes')
            cracks_img = analyzer.draw_analysis_results(filter_type='cracks')
            cracks_path, cracks_filename = save_result_image(cracks_img, 'cracks')
            grains_img = analyzer.draw_analysis_results(filter_type='grains')
            grains_path, grains_filename = save_result_image(grains_img, 'grains')
        elif analysis_type == 'holes':
            result_img = analyzer.draw_analysis_results(filter_type='holes')
            result_path, result_filename = save_result_image(result_img, 'all')
            holes_path = result_path
            cracks_path = result_path
            grains_path = result_path
        elif analysis_type == 'cracks':
            result_img = analyzer.draw_analysis_results(filter_type='cracks')
            result_path, result_filename = save_result_image(result_img, 'all')
            holes_path = result_path
            cracks_path = result_path
            grains_path = result_path
        elif analysis_type == 'grains':
            result_img = analyzer.draw_analysis_results(filter_type='grains')
            result_path, result_filename = save_result_image(result_img, 'all')
            holes_path = result_path
            cracks_path = result_path
            grains_path = result_path

        # 存储到session
        session['analysis_params'] = analysis_params
        session['result_image_path'] = result_path
        session['result_filename'] = result_filename
        session['result_image_holes'] = holes_path
        session['result_image_cracks'] = cracks_path
        session['result_image_grains'] = grains_path
        session['holes_count'] = len(holes)
        session['cracks_count'] = len(cracks)
        session['grains_count'] = len(grains)
        session['analysis_type'] = analysis_type

        # 存储详细结果（去掉contour以便JSON序列化）
        session['results'] = {
            'holes': [{'id': h['id'], 'area': h['area'], 'area_cm2': h['area_cm2'],
                       'diameter': h['diameter'], 'diameter_cm': h['diameter_cm'],
                       'position': h['position']} for h in holes],
            'cracks': [{'id': c['id'], 'length': c['length'], 'length_cm': c['length_cm'],
                        'width': c['width'], 'width_cm': c['width_cm'],
                        'aspect_ratio': c['aspect_ratio'], 'type': c['type'],
                        'position': c['position']} for c in cracks],
            'grains': [{'id': g['id'], 'area': g['area'], 'area_cm2': g['area_cm2'],
                        'size': g['size'], 'size_cm': g['size_cm'],
                        'circularity': g['circularity'], 'position': g['position']} for g in grains],
            'params': analysis_params
        }

        return jsonify({
            'success': True,
            'result_image': f"/results_folder/{result_filename}",
            'summary': {
                'holes': len(holes),
                'cracks': len(cracks),
                'grains': len(grains)
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/results')
def results():
    if 'results' not in session:
        return redirect(url_for('index'))

    results_data = session['results']

    # 确保 params 在 results 中
    if 'params' not in results_data:
        results_data['params'] = session.get('analysis_params', {
            'hole_threshold': 79, 'hole_min_area': 50,
            'crack_threshold': 97, 'crack_min_points': 10,
            'grain_threshold': 138, 'grain_min_area': 15
        })

    # 获取分析类型，决定默认展示的标签页
    analysis_type = session.get('analysis_type', 'all')
    # URL参数可以覆盖
    tab_param = request.args.get('tab', analysis_type)
    active_tab = tab_param if tab_param in ('all', 'holes', 'cracks', 'grains') else 'all'

    return render_template('results.html',
                          results=results_data,
                          result_image_path=session.get('result_image_path', ''),
                          result_image_holes=session.get('result_image_holes', ''),
                          result_image_cracks=session.get('result_image_cracks', ''),
                          result_image_grains=session.get('result_image_grains', ''),
                          image_path=session.get('image_path', ''),
                          active_tab=active_tab)


@app.route('/report')
def report():
    if 'results' not in session:
        return redirect(url_for('index'))

    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=72)

        styles = getSampleStyleSheet()

        # ========== 注册中文字体（支持 Windows/macOS/Linux）==========
        chinese_font = None
        
        # 根据操作系统选择字体路径
        if sys.platform == 'win32':
            # Windows 系统字体
            font_paths = [
                ('C:/Windows/Fonts/simsun.ttc', 'SimSun'),
                ('C:/Windows/Fonts/simhei.ttf', 'SimHei'),
                ('C:/Windows/Fonts/msyh.ttc', 'Microsoft YaHei'),
                ('C:/Windows/Fonts/kaiti.ttf', 'KaiTi'),
            ]
        elif sys.platform == 'darwin':
            # macOS 系统字体
            font_paths = [
                ('/System/Library/Fonts/STSong.ttf', 'STSong-Light'),
                ('/System/Library/Fonts/PingFang.ttc', 'PingFang SC'),
                ('/Library/Fonts/Arial Unicode.ttf', 'Arial Unicode MS'),
            ]
        else:
            # Linux 系统字体
            font_paths = [
                ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 'NotoSansCJK'),
                ('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc', 'WenQuanYiMicroHei'),
                ('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', 'WenQuanYi Zen Hei'),
                ('/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc', 'NotoSerifCJK'),
            ]
        
        # 尝试注册每个字体
        for font_path, font_name in font_paths:
            try:
                if os.path.exists(font_path):
                    pdfmetrics.registerFont(TTFont(font_name, font_path))
                    chinese_font = font_name
                    print(f"✅ 成功加载字体: {font_name} ({font_path})")
                    break
            except Exception as e:
                print(f"⚠️ 加载字体失败 {font_name}: {e}")
                continue
        
        # 如果都失败了，使用 Helvetica（会乱码，但至少不报错）
        if chinese_font is None:
            chinese_font = 'Helvetica'
            print("⚠️ 未找到中文字体，将使用默认字体（可能乱码）")

        # ========== 创建样式 ==========
        title_style = styles['Title'].clone('cn_title')
        title_style.fontName = chinese_font
        title_style.fontSize = 18
        title_style.leading = 22
        title_style.alignment = 1  # 居中

        heading_style = styles['Heading2'].clone('cn_heading')
        heading_style.fontName = chinese_font
        heading_style.fontSize = 14
        heading_style.leading = 18

        chinese_style = styles["Normal"].clone('chinese')
        chinese_style.fontName = chinese_font
        chinese_style.fontSize = 10
        chinese_style.leading = 14
        chinese_style.wordWrap = 'CJK'

        elements = []
        elements.append(Paragraph("海洋地质岩心分析报告", title_style))
        elements.append(Spacer(1, 15))

        info_table_data = [
            [Paragraph("<b>分析日期</b>", chinese_style),
             Paragraph(datetime.now().strftime('%Y-%m-%d %H:%M'), chinese_style)],
            [Paragraph("<b>样本图像</b>", chinese_style),
             Paragraph(os.path.basename(session.get('image_path', '')), chinese_style)]
        ]
        info_table = Table(info_table_data, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('FONT', (0,0), (-1,-1), chinese_font),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ]))
        elements.append(info_table)
        elements.append(Spacer(1, 20))

        # 添加分析结果图
        result_image_path = session.get('result_image_path', '')
        if result_image_path and os.path.exists(result_image_path):
            try:
                from PIL import Image as PILImage
                img = PILImage.open(result_image_path)
                img_w, img_h = img.size
                max_w = 6.5 * inch
                max_h = 4 * inch
                ratio = min(max_w / img_w, max_h / img_h, 1.0)
                draw_w = img_w * ratio
                draw_h = img_h * ratio
                img.close()
                img_element = RLImage(result_image_path, width=draw_w, height=draw_h)
                elements.append(img_element)
                elements.append(Spacer(1, 15))
            except Exception as img_err:
                pass
            
        results_data = session['results']

        if results_data.get('holes'):
            elements.append(Paragraph("孔洞分析", heading_style))
            elements.append(Spacer(1, 10))
            hole_data = [[Paragraph("ID", chinese_style), Paragraph("面积(cm²)", chinese_style),
                          Paragraph("直径(cm)", chinese_style), Paragraph("位置", chinese_style)]]
            for hole in results_data['holes']:
                hole_data.append([Paragraph(str(hole['id']), chinese_style),
                    Paragraph(f"{hole.get('area_cm2', 0):.2f}", chinese_style),
                    Paragraph(f"{hole.get('diameter_cm', 0):.2f}", chinese_style),
                    Paragraph(f"{hole['position'][0]}, {hole['position'][1]}", chinese_style)])
            hole_table = Table(hole_data, repeatRows=1)
            hole_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E6F2FF")),
                ('FONT', (0,0), (-1,-1), chinese_font), ('FONTSIZE', (0,0), (-1,-1), 10),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F5F9FF")]),
            ]))
            elements.append(hole_table)
            elements.append(Spacer(1, 20))

        if results_data.get('cracks'):
            elements.append(Paragraph("裂缝分析", heading_style))
            elements.append(Spacer(1, 10))
            crack_data = [[Paragraph("ID", chinese_style), Paragraph("长度(cm)", chinese_style),
                           Paragraph("宽度(cm)", chinese_style), Paragraph("类型", chinese_style)]]
            for crack in results_data['cracks']:
                crack_data.append([Paragraph(str(crack['id']), chinese_style),
                    Paragraph(f"{crack.get('length_cm', 0):.2f}", chinese_style),
                    Paragraph(f"{crack.get('width_cm', 0):.2f}", chinese_style),
                    Paragraph(crack.get('type', '未知'), chinese_style)])
            crack_table = Table(crack_data, repeatRows=1)
            crack_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E6F2FF")),
                ('FONT', (0,0), (-1,-1), chinese_font), ('FONTSIZE', (0,0), (-1,-1), 10),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F5F9FF")]),
            ]))
            elements.append(crack_table)
            elements.append(Spacer(1, 20))

        if results_data.get('grains'):
            elements.append(Paragraph("粒度分析", heading_style))
            elements.append(Spacer(1, 10))
            grain_data = [[Paragraph("ID", chinese_style), Paragraph("尺寸(cm)", chinese_style),
                           Paragraph("圆度", chinese_style), Paragraph("位置", chinese_style)]]
            for grain in results_data['grains']:
                grain_data.append([Paragraph(str(grain['id']), chinese_style),
                    Paragraph(f"{grain.get('size_cm', 0):.2f}", chinese_style),
                    Paragraph(f"{grain.get('circularity', 0):.2f}", chinese_style),
                    Paragraph(f"{grain['position'][0]}, {grain['position'][1]}", chinese_style)])
            grain_table = Table(grain_data, repeatRows=1)
            grain_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E6F2FF")),
                ('FONT', (0,0), (-1,-1), chinese_font), ('FONTSIZE', (0,0), (-1,-1), 10),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F5F9FF")]),
            ]))
            elements.append(grain_table)
            elements.append(Spacer(1, 20))

        elements.append(Paragraph("分析结论", heading_style))
        elements.append(Spacer(1, 10))
        conclusion = []
        if len(results_data.get('holes', [])) > 10:
            conclusion.append("样本存在显著孔洞结构")
        if len(results_data.get('cracks', [])) > 5:
            conclusion.append("发现明显裂缝网络")
        if len(results_data.get('grains', [])) > 50:
            conclusion.append("颗粒分布密集")
        conclusion_text = "；".join(conclusion) if conclusion else "未发现明显地质异常"
        elements.append(Paragraph(conclusion_text, chinese_style))

        doc.build(elements)
        buffer.seek(0)
        return send_file(buffer, as_attachment=True,
                         download_name=f"岩心分析报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                         mimetype='application/pdf')
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"PDF生成失败: {str(e)}", 500


@app.route('/results_folder/<filename>')
def results_folder(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)


@app.route('/uploads_folder/<filename>')
def uploads_folder(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/download_app')
def download_app():
    return send_file(__file__, as_attachment=True, download_name='app.py')


if __name__ == '__main__':
    app.run(debug=True)
