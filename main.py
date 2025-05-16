# -*- coding: utf-8 -*-
"""
Created on Fri May 16 15:37:50 2025

@author: ktrpt
"""
import streamlit as st
import cv2
import tempfile
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="Hand 3D Pose Viewer", layout="centered")
st.title("MediaPipe Hands 3D Visualization")

uploaded_file = st.file_uploader("🎥 Upload a video file (.mp4, .mov, .avi)", type=["mp4", "mov", "avi"])

if uploaded_file:
    # Save to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Video metadata
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    st.markdown(f"📊 Total Frames: `{frame_count}` | FPS: `{fps:.2f}` | Size: `{width}x{height}`")

    # Frame selection
    frame_idx = st.slider("Select Frame", 0, frame_count - 1, 0)

    # Preview selected frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret:
        preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(preview, caption=f"Frame {frame_idx}", use_container_width=True)

    if st.button("Process this frame"):
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        hands.close()

        if results.multi_hand_landmarks:
            for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                label = handedness.classification[0].label  # 'Left' or 'Right'
                corrected_label = "Right" if label == "Left" else "Left"  # Flip label for user-facing view

                st.subheader(f"Hand {idx + 1} ({corrected_label})")

                # Extract landmarks
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                df = pd.DataFrame(coords, columns=['x', 'y', 'z'])

                # Flip Z for better visualization
                df['z'] = -df['z']

                # 3D plot
                fig = plt.figure(figsize=(6, 6))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(df['x'], df['z'], df['y'], c='blue', s=50)

                for connection in mp_hands.HAND_CONNECTIONS:
                    p1, p2 = connection
                    ax.plot(
                        [df.iloc[p1]['x'], df.iloc[p2]['x']],
                        [df.iloc[p1]['z'], df.iloc[p2]['z']],
                        [df.iloc[p1]['y'], df.iloc[p2]['y']],
                        'gray'
                    )

                ax.set_xlabel("X")
                ax.set_ylabel("Z (Depth)")
                ax.set_zlabel("Y")
                ax.view_init(elev=10, azim=70)
                st.pyplot(fig)

                # CSV download
                st.markdown("### ✍️ Coordinate Data")
                st.dataframe(df)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"Download CSV for {corrected_label} Hand",
                    data=csv,
                    file_name=f"{corrected_label.lower()}_hand_frame_{frame_idx}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("No hand landmarks detected.")
