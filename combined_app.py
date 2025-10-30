from tkinter import *
from tkinter import messagebox, filedialog, ttk
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from playsound import playsound
import numpy as np
import cv2
import os
import threading
from PIL import Image, ImageTk
import base64
from io import BytesIO

class ModernEmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎵 Emotion-Based Music Recommender")
        self.root.geometry("900x800")
        self.root.configure(bg='#f6f6fa')
        
        # Global variables
        self.filename = ""
        self.faces = None
        self.frame = None
        self.current_image = None
        self.emotion = "None"
        
        # Load models
        self.detection_model_path = 'haarcascade_frontalface_default.xml'
        self.emotion_model_path = '_mini_XCEPTION.106-0.65.hdf5'
        self.face_detection = cv2.CascadeClassifier(self.detection_model_path)
        self.emotion_classifier = load_model(self.emotion_model_path, compile=False)
        self.EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        self.container = Frame(self.root, bg='#fff', padx=32, pady=40)
        self.container.pack(fill=BOTH, expand=True)
        
        # Title
        title = Label(self.container, 
                     text="🎵 Emotion-Based Music Recommender",
                     font=('Segoe UI', 24, 'bold'),
                     fg='#4b2996',
                     bg='#fff')
        title.pack(pady=(0, 16))
        
        # Section header
        section_header = Label(self.container,
                             text="How are you feeling today?",
                             font=('Segoe UI', 18, 'bold'),
                             fg='#222',
                             bg='#fff')
        section_header.pack(pady=(32, 18))
        
        # Main content frame
        content_frame = Frame(self.container, bg='#fff')
        content_frame.pack(fill=X, pady=10)
        
        # Preview box
        self.preview_frame = Frame(content_frame, 
                                 bg='#222',
                                 width=350,
                                 height=180,
                                 bd=0,
                                 highlightthickness=0)
        self.preview_frame.pack(side=LEFT, padx=(0, 32))
        self.preview_frame.pack_propagate(False)
        
        self.preview_label = Label(self.preview_frame, bg='#222')
        self.preview_label.pack(expand=True, fill=BOTH)
        
        # Upload section
        upload_frame = Frame(content_frame, bg='#fff')
        upload_frame.pack(side=LEFT, fill=Y, padx=(0, 32))
        
        upload_label = Label(upload_frame,
                           text="📁 Upload Image",
                           font=('Segoe UI', 12),
                           bg='#fff',
                           fg='#222')
        upload_label.pack(pady=(0, 12))
        
        self.upload_btn = Button(upload_frame,
                               text="Choose File",
                               font=('Segoe UI', 11),
                               command=self.upload_image,
                               bg='#fff',
                               fg='#4b2996',
                               bd=1,
                               relief=SOLID)
        self.upload_btn.pack(pady=(0, 12))
        
        self.analyze_btn = Button(upload_frame,
                                text="🔍 Analyze Emotion",
                                font=('Segoe UI', 11, 'bold'),
                                command=self.auto_analyze_image,
                                bg='#7c3aed',
                                fg='#fff',
                                bd=0,
                                padx=20,
                                pady=10)
        self.analyze_btn.pack(pady=(18, 0))
        
        # Emotion status
        self.emotion_label = Label(content_frame,
                                 text="Detected Emotion: None",
                                 font=('Segoe UI', 11),
                                 bg='#fff',
                                 fg='#222')
        self.emotion_label.pack(side=LEFT, fill=Y)
        
        # Songs section
        songs_header = Label(self.container,
                           text="Songs for your current mood:",
                           font=('Segoe UI', 18, 'bold'),
                           fg='#222',
                           bg='#fff')
        songs_header.pack(pady=(32, 18), anchor=W)
        
        # Song card
        self.song_card = Frame(self.container,
                             bg='#f8f7fc',
                             padx=24,
                             pady=24)
        self.song_card.pack(fill=X, pady=(0, 20))
        
        # Album cover
        self.album_cover = Label(self.song_card,
                               text="🎵",
                               font=('Segoe UI', 24),
                               bg='#fff',
                               fg='#bbb',
                               width=4,
                               height=2)
        self.album_cover.pack(side=LEFT, padx=(0, 24))
        
        # Song details
        details_frame = Frame(self.song_card, bg='#f8f7fc')
        details_frame.pack(side=LEFT, fill=Y)
        
        self.song_title = Label(details_frame,
                              text="Song Title",
                              font=('Segoe UI', 14, 'bold'),
                              bg='#f8f7fc',
                              fg='#222')
        self.song_title.pack(anchor=W)
        
        self.artist_name = Label(details_frame,
                               text="Artist Name",
                               font=('Segoe UI', 12),
                               bg='#f8f7fc',
                               fg='#666')
        self.artist_name.pack(anchor=W)
        
        # Play button
        self.play_btn = Button(self.song_card,
                             text="▶ Play",
                             font=('Segoe UI', 11, 'bold'),
                             command=self.play_song,
                             bg='#4b2996',
                             fg='#fff',
                             bd=0,
                             padx=20,
                             pady=10)
        self.play_btn.pack(side=RIGHT)
        
    def upload_image(self):
        self.filename = filedialog.askopenfilename(
            initialdir="images",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if self.filename:
            # Display image in preview
            img = Image.open(self.filename)
            img = img.resize((350, 180), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=photo)
            self.preview_label.image = photo
            self.auto_analyze_image()
            
    def auto_analyze_image(self):
        if not self.filename:
            messagebox.showwarning("Warning", "Please upload an image first.")
            return
            
        frame = cv2.imread(self.filename, 0)
        faces = self.face_detection.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) > 0:
            (fX, fY, fW, fH) = sorted(faces, reverse=True, 
                                    key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            roi = frame[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            preds = self.emotion_classifier.predict(roi)[0]
            label = self.EMOTIONS[preds.argmax()]
            self.emotion = label
            self.emotion_label.config(text=f"Detected Emotion: {label.title()}")
            self.show_song_for_emotion(label)
        else:
            self.emotion_label.config(text="Detected Emotion: None")
            messagebox.showinfo("Emotion Prediction", "No face detected in uploaded image.")
            
    def show_song_for_emotion(self, label):
        song_dir = os.path.join(os.path.dirname(__file__), 'songs')
        found_song = None
        
        for root, dirs, files in os.walk(song_dir):
            for file in files:
                if label.lower() in file.lower():
                    found_song = file
                    break
            if found_song:
                break
                
        if found_song:
            self.song_title.config(text=os.path.splitext(found_song)[0].title())
            self.artist_name.config(text="Unknown Artist")
            
            # Try to load album cover
            cover_path = os.path.join(os.path.dirname(__file__), 'images', f"{label}.png")
            if os.path.exists(cover_path):
                img = Image.open(cover_path)
                img = img.resize((60, 60), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.album_cover.config(image=photo, text="")
                self.album_cover.image = photo
            else:
                self.album_cover.config(image="", text="🎵")
                
            self.play_btn.song_path = os.path.join(song_dir, found_song)
        else:
            self.song_title.config(text="No song found")
            self.artist_name.config(text="")
            self.album_cover.config(image="", text="🎵")
            self.play_btn.song_path = None
            
    def play_song(self):
        song_path = getattr(self.play_btn, 'song_path', None)
        if song_path:
            threading.Thread(target=playsound, args=(song_path,), daemon=True).start()
        else:
            messagebox.showwarning("Warning", "No song selected.")

def main():
    root = Tk()
    app = ModernEmotionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 