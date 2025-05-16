# -*- coding: utf-8 -*-
"""
Created on Fri May 16 15:37:50 2025

@author: ktrpt
"""
import streamlit as st
import cv2
import tempfile
import numpy as np
import mediapipe as mp
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="3D Hand Landmark Viewer", layout="wide")
st.title("3D Hand Landmark Viewer with MediaPipe")

# MediaPipe Hands 初期化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,
                       max_num_hands=2,
                       model_complexity=1,
                       min_detection_confidence=0.5)

# 動画アップロード
uploaded_file = st.file_uploader("動画ファイルをアップロード", type=["mp4", "mov", "avi"])
if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    st.success(f"動画読み込み完了｜総フレーム数：{total_frames}")

    selected_frame = st.slider("表示するフレームを選択", 0, total_frames - 1, 0)
    show_right = st.checkbox("右手を表示", value=True)
    show_left = st.checkbox("左手を表示", value=True)

    # 指定フレーム取得
    cap.set(cv2.CAP_PROP_POS_FRAMES, selected_frame)
    ret, frame = cap.read()
    if ret:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        st.image(rgb, caption=f"Frame {selected_frame}", use_container_width=True)

        # ランドマーク検出と3D描画
        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[idx].classification[0].label
                if (handedness == "Right" and not show_right) or (handedness == "Left" and not show_left):
                    continue

                landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

                fig = plt.figure(figsize=(6, 6))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(landmark_array[:, 0], landmark_array[:, 1], landmark_array[:, 2], c='crimson', s=40)
                ax.set_xlabel("X (左右)")
                ax.set_ylabel("Y (上下)")
                ax.set_zlabel("Z (奥行)")
                ax.view_init(elev=20, azim=-60)
                ax.set_title(f"3D Hand - {handedness}")
                st.pyplot(fig)

                # CSV 出力
                df = pd.DataFrame(landmark_array, columns=["x", "y", "z"])
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"{handedness}download CSV",
                    data=csv,
                    file_name=f"{handedness}_hand_frame{selected_frame}.csv",
                    mime="text/csv"
                )
    else:
        st.error("error")
    cap.release()
