
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import tempfile
import datetime
import os
from PIL import Image
from ML_dassui import predict_dassui

st.set_page_config(page_title="脱水判定アプリ", layout="centered")
st.title("💧 指の腹画像で脱水判定")

mode = st.radio("モード選択", ("絶対モード", "相対モード"))
mode_key = "absolute" if mode == "絶対モード" else "relative"

uploaded_main = st.file_uploader("判定対象画像をアップロード", type=["jpg", "jpeg", "png", "bmp", "gif"])
uploaded_baseline = None
if mode_key == "relative":
    uploaded_baseline = st.file_uploader("基準画像（非脱水時）をアップロード", type=["jpg", "jpeg", "png", "bmp", "gif"], key="baseline")

if uploaded_main:
    st.image(uploaded_main, caption="判定画像プレビュー", use_column_width=True)

if st.button("評価"):
    if not uploaded_main:
        st.warning("判定画像をアップロードしてください。")
    elif mode_key == "relative" and not uploaded_baseline:
        st.warning("相対モードでは基準画像が必要です。")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tf_main:
            tf_main.write(uploaded_main.read())
            main_path = tf_main.name

        baseline_path = None
        if mode_key == "relative" and uploaded_baseline:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tf_base:
                tf_base.write(uploaded_baseline.read())
                baseline_path = tf_base.name

        label, score, heatmap = predict_dassui(main_path, mode_key, baseline_path)
        st.success(f"推定脱水度: {label}（{score:.1f}%）")
        st.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), caption="ハイライト画像", use_column_width=True)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        df = pd.DataFrame([{"filename": uploaded_main.name, "mode": mode_key, "result": label, "score": score}])
        csv_path = os.path.join(os.path.expanduser("~"), f"Dassui_{ts}.csv")
        df.to_csv(csv_path, index=False)
        st.info(f"結果を保存しました: {csv_path}")
