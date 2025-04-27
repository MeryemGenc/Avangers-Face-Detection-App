import streamlit as st
import cv2
import tempfile
import numpy as np
from deepface import DeepFace
import os
from datetime import datetime
import shutil

info_text = """\
✨ Merhaba! ✨

⚡ Dikkat: Görüntünün net ve aydınlık olması tanımanın doğruluğunu artırır. Karakterlerin yüzü doğrudan ve net görünmelidir. Maskeli, karanlık veya bulanık sahneler tanımayı zorlaştırabilir.
"""

db_path = "known_faces"

def detect_and_display(img, name, analysis_cache):
    region = analysis_cache.get("region", None)

    if region:
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    y_offset = 50 
    gender_info = analysis_cache.get("gender", {})
    if isinstance(gender_info, dict):
        woman_score = gender_info.get("Woman", 0)
        man_score = gender_info.get("Man", 0)
        detected_gender = "Woman" if woman_score > man_score else "Man"
    else:
        detected_gender = "Unknown"
    display_texts = [
        f"Name: {name}",
        f"Age: {analysis_cache.get('age', '...')}",
        f"Gender: {detected_gender}",
        f"Emotion: {analysis_cache.get('emotion', '...')}"
    ]

    for text in display_texts:
        cv2.putText(img, text, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        y_offset += 30

    return img

def process_video(uploaded_video, db_path):
    # Geçici dosya 
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "uploaded_video.mp4")
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_video.read())

    cap = cv2.VideoCapture(temp_path)
    
    if not cap.isOpened():
        st.error("Video açılamadı!")
        shutil.rmtree(temp_dir)
        return

    # Video yazıcı ayarları
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Kaydedilecek video dosya adı
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"result_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    name = "Unknown"
    analysis_cache = {}

    stframe = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 15 == 0:
            try:
                result = DeepFace.find(frame, db_path=db_path, model_name="Facenet", 
                detector_backend="opencv", enforce_detection=True)
                
                if len(result) > 0:
                    first_match = result[0]
                    if "identity" in first_match and len(first_match["identity"]) > 0:
                        identity_path = first_match["identity"].iloc[0]
                        folder_name = os.path.basename(os.path.dirname(identity_path))
                        name = folder_name
                    else:
                        name = "Unknown"
                else:
                    name = "Unknown"

                face_data = DeepFace.analyze(frame, actions=['age', 'gender', 'emotion'], enforce_detection=False)
                if isinstance(face_data, list) and len(face_data) > 0:
                    face = face_data[0]
                    analysis_cache = {
                        "age": face["age"],
                        "gender": face["gender"],
                        "emotion": face["dominant_emotion"],
                        "region": face["region"]
                    }
            except Exception as e:
                print(f"Hata oluştu: {e}")
                name = "Unknown"
                analysis_cache = {}

        processed_frame = detect_and_display(frame.copy(), name, analysis_cache)
        
        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)
        out.write(processed_frame)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = min(frame_count / total_frames, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"İşleniyor... %{int(progress*100)} Tamamlandı")
        
        frame_count += 1

    # Kaynakları serbest bırakma
    cap.release()
    out.release()
    
    # Geçici dosyaları temizleme
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Geçici dosya silinirken hata: {e}")
 
    status_text.text("✔️ İşlem tamamlandı!")
    progress_bar.empty()
    
    # İndirme butonunu 
    if os.path.exists(output_filename):
        with open(output_filename, "rb") as file:
            st.download_button(
                label="İşlenmiş Videoyu İndir",
                data=file,
                file_name=output_filename,
                mime="video/mp4"
            )
        os.remove(output_filename)

# Streamlit arayüz
st.title("Avengers Character Recognition")
st.markdown(info_text)

uploaded_video = st.file_uploader("Bir video dosyası yükle", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    process_video(uploaded_video, db_path)


