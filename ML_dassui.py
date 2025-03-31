
import cv2
import numpy as np
import joblib

model = joblib.load("dassui_model.pkl")

def extract_features(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (100, 100))
    mean = np.mean(resized)
    std = np.std(resized)
    contrast = resized.std()
    hist = cv2.calcHist([resized], [0], None, [16], [0, 256]).flatten()
    hist /= hist.sum()
    features = [mean, std, contrast] + hist.tolist()
    return np.array(features)

def predict_dassui(image_path, mode="absolute", baseline_path=None):
    if mode == "absolute":
        X = extract_features(image_path).reshape(1, -1)
    elif mode == "relative" and baseline_path:
        feat1 = extract_features(baseline_path)
        feat2 = extract_features(image_path)
        X = (feat2 - feat1).reshape(1, -1)
    else:
        raise ValueError("モードが不正、または相対モードにおいて基準画像が未指定です。")

    score = model.predict_proba(X)[0][1] * 100
    label = "低" if score < 30 else "中" if score < 60 else "高"
    heatmap = highlight_region(image_path)
    return label, score, heatmap

def highlight_region(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    heatmap = np.uint8(255 * (laplacian - laplacian.min()) / (laplacian.ptp() + 1e-5))
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
    return overlay
