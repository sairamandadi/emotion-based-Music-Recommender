from tkinter import *
from tkinter import messagebox, filedialog, ttk
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from playsound import playsound
import numpy as np
import cv2
import os
import threading
import time
from PIL import Image, ImageTk
import pygame
import random

class ModernEmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion-Based Music Recommender")
        self.root.geometry("900x800")
        self.root.configure(bg='#f6f6fa')

        self.filename = ""
        self.emotion = "None"

        # Add variables to store last detected facial expression
        self.last_detected_emotion = None
        self.last_detected_confidence = 0
        self.last_emotion_frame = None
        
        # Animation related variables
        self.animation_active = False
        self.animation_frame_id = None
        self.equalizer_bars = []
        self.bar_heights = []
        self.bar_directions = []
        
        # Album cover animation variables
        self.album_pulse_active = False
        self.album_pulse_id = None
        self.album_scale = 1.0
        self.album_scale_direction = 0.01
        
        # Transition animation variables
        self.transition_active = False
        self.transition_alpha = 0
        self.transition_id = None
        
        # Emotion detection animation variables
        self.detection_animation_active = False
        self.detection_animation_id = None
        self.detection_dots = []
        self.detection_angle = 0

        # Background video setup - removed
        
        # Load models
        self.detection_model_path = 'haarcascade_frontalface_default.xml'
        self.emotion_model_path = '_mini_XCEPTION.106-0.65.hdf5'
        self.face_detection = cv2.CascadeClassifier(self.detection_model_path)
        self.emotion_classifier = load_model(self.emotion_model_path, compile=False)
        self.EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

        self.webcam_active = False
        self.cap = None

        pygame.mixer.init()
        self.currently_playing = None
        
        # Add playback state variables
        self.is_paused = False
        self.current_volume = 0.7  # Default volume (70%)
        pygame.mixer.music.set_volume(self.current_volume)
        self.is_muted = False
        self.previous_volume = self.current_volume
        self.progress_update_id = None
        
        # Add mode state variables
        self.shuffle_mode = False
        self.repeat_mode = False  # 0 = off, 1 = repeat one, 2 = repeat all
        self.repeat_one = False
        self.end_of_song_threshold = 0.98  # Consider song as ended when reaching 98% of duration
        
        # Add keyboard bindings for playback control
        self.root.bind("<space>", lambda event: self.toggle_pause())
        self.root.bind("<Left>", lambda event: self.previous_song())
        self.root.bind("<Right>", lambda event: self.next_song())
        self.root.bind("<Up>", lambda event: self.increase_volume())
        self.root.bind("<Down>", lambda event: self.decrease_volume())
        self.root.bind("m", lambda event: self.toggle_mute())
        self.root.bind("s", lambda event: self.toggle_shuffle())
        self.root.bind("r", lambda event: self.toggle_repeat())

        self.setup_ui()

    def setup_ui(self):
        # --- Create a canvas and add scrollbars ---
        self.canvas = Canvas(self.root, bg='#fff', highlightthickness=0)
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)

        # Vertical scrollbar
        v_scroll = Scrollbar(self.root, orient=VERTICAL, command=self.canvas.yview)
        v_scroll.pack(side=RIGHT, fill=Y)
        self.canvas.configure(yscrollcommand=v_scroll.set)

        # Horizontal scrollbar
        h_scroll = Scrollbar(self.root, orient=HORIZONTAL, command=self.canvas.xview)
        h_scroll.pack(side=BOTTOM, fill=X)
        self.canvas.configure(xscrollcommand=h_scroll.set)

        # --- Frame inside the canvas ---
        self.container = Frame(self.canvas, bg='#fff', padx=32, pady=40)
        self.container_id = self.canvas.create_window((0, 0), window=self.container, anchor='nw')

        # --- Bindings to make scrolling work and resize properly ---
        self.container.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Header frame with title and language selection
        header_frame = Frame(self.container, bg='#fff')
        header_frame.pack(fill=X, pady=(0, 16))

        # Title on the left side
        Label(header_frame, text="Emotion-Based Music Recommender",
              font=('Segoe UI', 24, 'bold'), fg='#4b2996', bg='#fff').pack(side=LEFT)

        # Language selection on the right
        lang_frame = Frame(header_frame, bg='#fff')
        lang_frame.pack(side=RIGHT, padx=10)
        
        # Add label for language selector
        self.language_label = Label(lang_frame, text="Select Language:", font=('Segoe UI', 12), 
                              bg='#fff', fg='#8e8e8e')  # Start with gray text to indicate disabled
        self.language_label.pack(side=LEFT)
        
        # Language dropdown configuration
        self.language_var = StringVar(value="english")
        languages = [
            "english", "hindi", "tamil", "telugu", "bengali", 
            "gujarati", "german", "japanese", "kannada", 
            "malayalam", "marathi", "odia", "russian", "spanish"
        ]
        self.language_menu = OptionMenu(lang_frame, self.language_var, *languages)
        self.language_menu.config(font=('Segoe UI', 12), bg='#fff')
        self.language_menu.pack(side=LEFT, padx=10)

        # Initially disable the language menu
        self.language_menu.config(state=DISABLED)
        
        # Track changes to the language dropdown for immediate updates
        self.language_var.trace('w', lambda *args: self.on_language_change())

        Label(self.container, text="How are you feeling today?",
              font=('Segoe UI', 18, 'bold'), fg='#222', bg='#fff').pack(pady=(32, 18))

        content_frame = Frame(self.container, bg='#fff')
        content_frame.pack(fill=X, pady=10)

        self.preview_frame = Frame(content_frame, bg='#222', width=350, height=180)
        self.preview_frame.pack(side=LEFT, padx=(0, 32))
        self.preview_frame.pack_propagate(False)

        self.preview_label = Label(self.preview_frame, bg='#222')
        self.preview_label.pack(expand=True, fill=BOTH)

        upload_frame = Frame(content_frame, bg='#fff')
        upload_frame.pack(side=LEFT, fill=Y, padx=(0, 32))

        # Add a row of buttons at the top
        top_buttons_frame = Frame(upload_frame, bg='#fff')
        top_buttons_frame.pack(fill=X, pady=(0, 15))

        # Upload Image button with hover effect
        upload_btn = Button(top_buttons_frame, text="Upload Image", font=('Segoe UI', 11),
               command=self.upload_image, bg='#fff', fg='#4b2996', bd=1, relief=SOLID)
        upload_btn.pack(side=LEFT, padx=(0, 10))
        self.add_button_hover_effect(upload_btn)

        # Enter Text button with hover effect
        text_btn = Button(top_buttons_frame, text="Enter Text", font=('Segoe UI', 11),
               command=self.show_emotion_selector, bg='#fff', fg='#4b2996', bd=1, relief=SOLID)
        text_btn.pack(side=LEFT)
        self.add_button_hover_effect(text_btn)

        analyze_btn = Button(upload_frame, text="Analyze Emotion", font=('Segoe UI', 11, 'bold'),
            command=self.analyze_emotion, bg='#7c3aed', fg='#fff', bd=0,
            padx=20, pady=10)
        analyze_btn.pack(pady=(18, 18))
        self.add_button_hover_effect(analyze_btn, hover_bg='#9361ff')

        self.webcam_btn = Button(upload_frame, text="Your Current Mood", font=('Segoe UI', 11, 'bold'),
                        command=self.toggle_webcam, bg='#4b2996', fg='#fff', bd=0,
                        padx=20, pady=10)
        self.webcam_btn.pack(pady=(0, 12))
        self.add_button_hover_effect(self.webcam_btn, hover_bg='#6039c0')

        self.emotion_label = Label(content_frame, text="Detected Emotion: None",
                                   font=('Segoe UI', 11), bg='#fff', fg='#222')
        self.emotion_label.pack(side=LEFT, fill=Y)

        self.song_card = Frame(self.container, bg='#f8f7fc', padx=24, pady=24)
        self.song_card.pack(fill=X, pady=(0, 20))

        # Create animation canvas on the left side of song card
        self.animation_canvas = Canvas(self.song_card, width=40, height=60, bg='#f8f7fc', highlightthickness=0)
        self.animation_canvas.pack(side=LEFT, padx=(0, 5))
        
        # Create the equalizer bars
        self.create_equalizer_bars()

        self.album_cover = Label(self.song_card, text="Music", font=('Segoe UI', 24),
                                 bg='#fff', fg='#bbb', width=4, height=2)
        self.album_cover.pack(side=LEFT, padx=(5, 24))

        details_frame = Frame(self.song_card, bg='#f8f7fc')
        details_frame.pack(side=LEFT, fill=Y)

        self.song_title = Label(details_frame, text="Song Title",
                                font=('Segoe UI', 14, 'bold'), bg='#f8f7fc', fg='#222')
        self.song_title.pack(anchor=W)

        self.artist_name = Label(details_frame, text="Artist Name",
                                 font=('Segoe UI', 12), bg='#f8f7fc', fg='#666')
        self.artist_name.pack(anchor=W)

        # Song control buttons
        controls_frame = Frame(self.song_card, bg='#f8f7fc')
        controls_frame.pack(side=RIGHT, padx=10)

        # Create playback buttons frame
        playback_buttons_frame = Frame(controls_frame, bg='#f8f7fc')
        playback_buttons_frame.grid(row=0, column=0, columnspan=5, pady=(0, 5))

        # Previous song button
        self.prev_btn = Button(playback_buttons_frame, text="⏮", font=('Segoe UI', 11, 'bold'),
                          command=self.previous_song, bg='#4b2996', fg='#fff', bd=0, padx=8, pady=5)
        self.prev_btn.grid(row=0, column=0, padx=2)
        self.add_button_hover_effect(self.prev_btn, hover_bg='#6039c0')

        self.play_btn = Button(playback_buttons_frame, text="▶", font=('Segoe UI', 11, 'bold'),
                               command=self.play_song, bg='#4b2996', fg='#fff', bd=0, padx=8, pady=5)
        self.play_btn.grid(row=0, column=1, padx=2)
        self.add_button_hover_effect(self.play_btn, hover_bg='#6039c0')

        self.pause_btn = Button(playback_buttons_frame, text="⏸", font=('Segoe UI', 11, 'bold'),
                              command=self.toggle_pause, bg='#4b2996', fg='#fff', bd=0, padx=8, pady=5)
        self.pause_btn.grid(row=0, column=2, padx=2)
        self.add_button_hover_effect(self.pause_btn, hover_bg='#6039c0')

        self.stop_btn = Button(playback_buttons_frame, text="⏹", font=('Segoe UI', 11, 'bold'),
                               command=self.stop_current_song, bg='#4b2996', fg='#fff', bd=0, padx=8, pady=5)
        self.stop_btn.grid(row=0, column=3, padx=2)
        self.add_button_hover_effect(self.stop_btn, hover_bg='#6039c0')

        self.next_btn = Button(playback_buttons_frame, text="⏭", font=('Segoe UI', 11, 'bold'),
                               command=self.next_song, bg='#4b2996', fg='#fff', bd=0, padx=8, pady=5)
        self.next_btn.grid(row=0, column=4, padx=2)
        self.add_button_hover_effect(self.next_btn, hover_bg='#6039c0')

        # Second row of control buttons
        additional_buttons_frame = Frame(controls_frame, bg='#f8f7fc')
        additional_buttons_frame.grid(row=1, column=0, columnspan=5, pady=(0, 5))

        self.shuffle_btn = Button(additional_buttons_frame, text="Shuffle", font=('Segoe UI', 9, 'bold'),
                              command=self.toggle_shuffle, bg='#4b2996', fg='#fff', bd=0, padx=8, pady=5)
        self.shuffle_btn.grid(row=0, column=0, padx=2)
        self.add_button_hover_effect(self.shuffle_btn, hover_bg='#6039c0')

        self.repeat_btn = Button(additional_buttons_frame, text="Repeat", font=('Segoe UI', 9, 'bold'),
                              command=self.toggle_repeat, bg='#4b2996', fg='#fff', bd=0, padx=8, pady=5)
        self.repeat_btn.grid(row=0, column=1, padx=2)
        self.add_button_hover_effect(self.repeat_btn, hover_bg='#6039c0')

        self.mute_btn = Button(additional_buttons_frame, text="Mute", font=('Segoe UI', 9, 'bold'),
                            command=self.toggle_mute, bg='#4b2996', fg='#fff', bd=0, padx=8, pady=5)
        self.mute_btn.grid(row=0, column=2, padx=2)
        self.add_button_hover_effect(self.mute_btn, hover_bg='#6039c0')

        # Add progress bar frame
        progress_frame = Frame(self.song_card, bg='#f8f7fc')
        progress_frame.pack(fill=X, padx=24, pady=(10, 0))

        # Time labels and progress bar
        self.current_time_label = Label(progress_frame, text="0:00", bg='#f8f7fc', fg='#666', font=('Segoe UI', 9))
        self.current_time_label.pack(side=LEFT, padx=(0, 5))

        # Create a frame to hold the progress bar to handle clicks
        self.progress_frame_container = Frame(progress_frame, bg='#f8f7fc')
        self.progress_frame_container.pack(side=LEFT, fill=X, expand=True)

        self.progress_bar = ttk.Progressbar(self.progress_frame_container, orient=HORIZONTAL, length=200, mode='determinate')
        self.progress_bar.pack(fill=X, expand=True)
        
        # Bind click events for seek functionality
        self.progress_frame_container.bind("<Button-1>", self.seek)
        self.progress_bar.bind("<Button-1>", self.seek)

        # Make the total time display more prominent
        self.total_time_label = Label(progress_frame, text="0:00", bg='#f8f7fc', fg='#333', font=('Segoe UI', 9, 'bold'))
        self.total_time_label.pack(side=LEFT, padx=(5, 0))

        # Volume slider frame
        volume_frame = Frame(self.song_card, bg='#f8f7fc')
        volume_frame.pack(fill=X, padx=24, pady=(5, 0))

        # Volume indicator with icon
        self.volume_icon_label = Label(volume_frame, text="Vol:", bg='#f8f7fc', fg='#4b2996', font=('Segoe UI', 10, 'bold'))
        self.volume_icon_label.pack(side=LEFT)
        
        self.volume_var = DoubleVar(value=self.current_volume*100)
        self.volume_slider = ttk.Scale(volume_frame, from_=0, to=100, orient=HORIZONTAL, 
                                   variable=self.volume_var, command=self.set_volume)
        self.volume_slider.pack(side=LEFT, fill=X, expand=True, padx=5)

        self.volume_value_label = Label(volume_frame, text=f"{int(self.current_volume*100)}%", 
                                    bg='#f8f7fc', fg='#666', font=('Segoe UI', 9))
        self.volume_value_label.pack(side=LEFT, padx=(0, 5))
        
        # Add volume buttons
        volume_buttons_frame = Frame(volume_frame, bg='#f8f7fc')
        volume_buttons_frame.pack(side=LEFT)
        
        self.vol_down_btn = Button(volume_buttons_frame, text="−", font=('Segoe UI', 11, 'bold'),
                              command=self.decrease_volume, bg='#e0e0e6', fg='#555', bd=0, padx=5, pady=0)
        self.vol_down_btn.pack(side=LEFT, padx=2)
        
        self.vol_up_btn = Button(volume_buttons_frame, text="+", font=('Segoe UI', 11, 'bold'),
                            command=self.increase_volume, bg='#e0e0e6', fg='#555', bd=0, padx=5, pady=0)
        self.vol_up_btn.pack(side=LEFT, padx=2)

        # Track information panel
        self.track_info_frame = Frame(self.container, bg='#f0f0f8', padx=15, pady=15)
        self.track_info_frame.pack(fill=X, pady=10, padx=40)
        
        # Header for track info
        track_info_header = Frame(self.track_info_frame, bg='#f0f0f8')
        track_info_header.pack(fill=X, pady=(0, 10))
        
        Label(track_info_header, text="Track Information", font=('Segoe UI', 12, 'bold'), 
             bg='#f0f0f8', fg='#4b2996').pack(side=LEFT)
        
        self.show_details_var = BooleanVar(value=False)
        self.show_details_btn = ttk.Checkbutton(track_info_header, text="Show Details", 
                                          variable=self.show_details_var, 
                                          command=self.toggle_track_details)
        self.show_details_btn.pack(side=RIGHT)
        
        # Basic track info (always visible)
        self.basic_info_frame = Frame(self.track_info_frame, bg='#f0f0f8')
        self.basic_info_frame.pack(fill=X)
        
        # Two columns for information
        left_column = Frame(self.basic_info_frame, bg='#f0f0f8')
        left_column.pack(side=LEFT, fill=Y, expand=True)
        
        right_column = Frame(self.basic_info_frame, bg='#f0f0f8')
        right_column.pack(side=RIGHT, fill=Y, expand=True)
        
        # Left column info
        self.info_title = Label(left_column, text="Title: Not Playing", font=('Segoe UI', 10),
                             bg='#f0f0f8', fg='#333', anchor='w', justify=LEFT)
        self.info_title.pack(fill=X, pady=2, anchor='w')
        
        self.info_artist = Label(left_column, text="Artist: Unknown", font=('Segoe UI', 10),
                              bg='#f0f0f8', fg='#333', anchor='w', justify=LEFT)
        self.info_artist.pack(fill=X, pady=2, anchor='w')
        
        self.info_emotion = Label(left_column, text="Emotion: None", font=('Segoe UI', 10),
                               bg='#f0f0f8', fg='#333', anchor='w', justify=LEFT)
        self.info_emotion.pack(fill=X, pady=2, anchor='w')
        
        # Right column info
        self.info_format = Label(right_column, text="Format: Unknown", font=('Segoe UI', 10),
                              bg='#f0f0f8', fg='#333', anchor='w', justify=LEFT)
        self.info_format.pack(fill=X, pady=2, anchor='w')
        
        self.info_duration = Label(right_column, text="Duration: --:--", font=('Segoe UI', 10),
                                bg='#f0f0f8', fg='#333', anchor='w', justify=LEFT)
        self.info_duration.pack(fill=X, pady=2, anchor='w')
        
        self.info_size = Label(right_column, text="Size: Unknown", font=('Segoe UI', 10),
                            bg='#f0f0f8', fg='#333', anchor='w', justify=LEFT)
        self.info_size.pack(fill=X, pady=2, anchor='w')
        
        # Detailed info frame (hidden by default)
        self.detailed_info_frame = Frame(self.track_info_frame, bg='#f0f0f8')
        self.detailed_info_frame.pack(fill=X, pady=(10, 0))
        self.detailed_info_frame.pack_forget()  # Hide initially
        
        self.info_path = Label(self.detailed_info_frame, text="Path: ", font=('Segoe UI', 9),
                            bg='#f0f0f8', fg='#555', wraplength=600, anchor='w', justify=LEFT)
        self.info_path.pack(fill=X, pady=2)
        
        self.info_bitrate = Label(self.detailed_info_frame, text="Bitrate: Unknown", font=('Segoe UI', 9),
                              bg='#f0f0f8', fg='#555', anchor='w', justify=LEFT)
        self.info_bitrate.pack(fill=X, pady=2)
        
        self.info_sample_rate = Label(self.detailed_info_frame, text="Sample Rate: Unknown", font=('Segoe UI', 9),
                                  bg='#f0f0f8', fg='#555', anchor='w', justify=LEFT)
        self.info_sample_rate.pack(fill=X, pady=2)
        
        self.info_channels = Label(self.detailed_info_frame, text="Channels: Unknown", font=('Segoe UI', 9),
                               bg='#f0f0f8', fg='#555', anchor='w', justify=LEFT)
        self.info_channels.pack(fill=X, pady=2)
        
        # Initially hide the track info until a song is playing
        self.track_info_frame.pack_forget()

        # Listbox for all matching songs with scrollbar on the left
        song_listbox_frame = Frame(self.container, bg='#fff')
        song_listbox_frame.pack(padx=40, pady=(0, 20), fill=X)

        scrollbar = Scrollbar(song_listbox_frame, orient=VERTICAL)
        scrollbar.pack(side=LEFT, fill=Y)

        self.song_listbox = Listbox(song_listbox_frame, font=('Segoe UI', 12), width=60, height=6, 
                                   yscrollcommand=scrollbar.set, cursor="hand2", 
                                   selectbackground="#4b2996", selectforeground="white")
        self.song_listbox.pack(side=LEFT, fill=BOTH, expand=True)

        scrollbar.config(command=self.song_listbox.yview)

        # Bind click and double-click for song selection
        self.song_listbox.bind('<<ListboxSelect>>', self.on_song_select)
        self.song_listbox.bind('<Double-1>', self.on_song_select)  # Double-click also works
        
        self.song_paths = []  # To store full paths of listed songs

    def _on_canvas_configure(self, event):
        # Resize the inner frame to match the canvas width
        self.canvas.itemconfig(self.container_id, width=event.width)

    def upload_image(self):
        self.filename = filedialog.askopenfilename(
            initialdir="images",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if self.filename:
            img = Image.open(self.filename)
            img = img.resize((350, 180), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=photo)
            self.preview_label.image = photo
            self.auto_analyze_image()

    def analyze_emotion(self):
        """Analyze emotion from either uploaded image or webcam based on current state"""
        # If we have an uploaded image, analyze it
        if self.filename:
            self.emotion_label.config(text="Analyzing uploaded image...")
            # Start the emotion detection animation
            self.start_detection_animation()
            # Schedule the actual analysis to allow animation to start
            self.root.after(100, self.auto_analyze_image)
        else:
            # No image uploaded, use webcam
            self.emotion_label.config(text="Initializing webcam for analysis...")
            # Start the emotion detection animation
            self.start_detection_animation()
            # Schedule the webcam analysis to allow animation to start
            self.root.after(100, self.analyze_with_webcam)

    def auto_analyze_image(self):
        """Analyze emotion from an uploaded image and recommend songs"""
        if not self.filename:
            messagebox.showwarning("Warning", "Please upload an image first.")
            self.stop_detection_animation()
            return

        image = cv2.imread(self.filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # Improve detection with histogram equalization
        
        faces = self.face_detection.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=8,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) > 0:
            (fX, fY, fW, fH) = sorted(faces, reverse=True,
                                      key=lambda x: x[2] * x[3])[0]
            
            # Expand ROI slightly for better emotion detection
            y_offset = int(fH * 0.1)
            x_offset = int(fW * 0.05)
            y1 = max(0, fY - y_offset)
            y2 = min(gray.shape[0], fY + fH + y_offset)
            x1 = max(0, fX - x_offset)
            x2 = min(gray.shape[1], fX + fW + x_offset)
            
            roi = gray[y1:y2, x1:x2]
            roi = cv2.resize(roi, (48, 48))
            roi = cv2.GaussianBlur(roi, (5, 5), 0)  # Reduce noise
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = self.emotion_classifier.predict(roi)[0]
            confidence = preds.max() * 100
            label = self.EMOTIONS[preds.argmax()]
            self.emotion = label
            
            # Stop the detection animation
            self.stop_detection_animation()
            
            # Display the confidence level
            self.emotion_label.config(text=f"Detected Emotion: {label.title()} ({confidence:.1f}%)")
            
            # Show the emotion on the image
            display_img = cv2.imread(self.filename)
            cv2.rectangle(display_img, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)
            emotion_text = f"{label.title()} ({confidence:.1f}%)"
            cv2.putText(display_img, emotion_text, (fX, fY - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Convert back for display
            display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(display_img)
            img = img.resize((350, 180), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=photo)
            self.preview_label.image = photo
            
            # Recommend songs based on detected emotion
            self.show_song_for_emotion(label)
        else:
            # Stop the detection animation
            self.stop_detection_animation()
            self.emotion_label.config(text="Detected Emotion: None")
            messagebox.showinfo("Emotion Prediction", "No face detected in uploaded image.")

    def show_song_for_emotion(self, label):
        """Show songs for the detected emotion and activate language selection"""
        # Activate the language menu
        self.activate_language_menu()
        
        selected_language = self.language_var.get()
        base_song_directory = os.path.join(os.path.dirname(__file__), 'songs', selected_language)
        
        # Create base songs directory if it doesn't exist
        if not os.path.exists(base_song_directory):
            try:
                os.makedirs(base_song_directory, exist_ok=True)
                print(f"Created songs directory: {base_song_directory}")
            except Exception as e:
                print(f"Error creating songs directory: {str(e)}")
        
        # Look for a folder with the emotion name
        emotion_folder = os.path.join(base_song_directory, label.lower())
        
        # Create emotion folder if it doesn't exist
        if not os.path.exists(emotion_folder):
            try:
                os.makedirs(emotion_folder, exist_ok=True)
                print(f"Created emotion folder: {emotion_folder}")
            except Exception as e:
                print(f"Error creating emotion folder: {str(e)}")
        
        found_songs = []

        # First try to find songs in the specific emotion folder
        if os.path.exists(emotion_folder):
            print(f"Looking for songs in emotion folder: {emotion_folder}")
            for file in os.listdir(emotion_folder):
                file_path = os.path.join(emotion_folder, file)
                if os.path.isfile(file_path) and file.lower().endswith(('.mp3', '.wav', '.ogg')):
                    found_songs.append(file_path)
        
        # If no songs found in emotion folder, search by filename as fallback
        if not found_songs:
            print(f"No songs found in emotion folder, searching by filename")
            for dir_path, dirs, files in os.walk(base_song_directory):
                for file in files:
                    if file.lower().endswith(('.mp3', '.wav', '.ogg')) and label.lower() in file.lower():
                        found_songs.append(os.path.join(dir_path, file))

        # Update the listbox
        self.song_listbox.delete(0, END)
        self.song_paths = found_songs
        for song_path in found_songs:
            self.song_listbox.insert(END, os.path.splitext(os.path.basename(song_path))[0].title())

        # Show first song in card, or clear if none
        if found_songs:
            first_song = found_songs[0]
            song_filename = os.path.splitext(os.path.basename(first_song))[0]
            
            # Try to extract artist name from filename (assume format: Artist - Title)
            artist_name = "Unknown Artist"
            if " - " in song_filename:
                parts = song_filename.split(" - ", 1)
                if len(parts) == 2:
                    artist_name = parts[0].title()
                    song_title = parts[1].title()
                else:
                    song_title = song_filename.title()
            else:
                # Alternative format: Title by Artist
                if " by " in song_filename.lower():
                    parts = song_filename.lower().split(" by ", 1)
                    if len(parts) == 2:
                        song_title = parts[0].title()
                        artist_name = parts[1].title()
                    else:
                        song_title = song_filename.title()
                else:
                    song_title = song_filename.title()
            
            # Update song info in display
            self.song_title.config(text=song_title)
            self.artist_name.config(text=artist_name)
            print(f"Song title: {song_title}, Artist: {artist_name}")
            
            # Check for emotion image and create images directory if needed
            images_dir = os.path.join(os.path.dirname(__file__), 'images')
            if not os.path.exists(images_dir):
                try:
                    os.makedirs(images_dir, exist_ok=True)
                    print(f"Created images directory: {images_dir}")
                except Exception as e:
                    print(f"Error creating images directory: {str(e)}")
            
            # Try to load emotion image
            cover_path = os.path.join(images_dir, f"{label}.png")
            
            # If the emotion image doesn't exist, create a colored placeholder
            if not os.path.exists(cover_path):
                try:
                    # Create a colored square based on emotion
                    colors = {
                        "happy": "#ffd166",    # Yellow
                        "sad": "#118ab2",      # Blue
                        "angry": "#ef476f",    # Red
                        "scared": "#7209b7",   # Purple
                        "surprised": "#06d6a0", # Teal
                        "neutral": "#8d99ae",  # Gray-Blue
                        "disgust": "#6a994e"   # Green
                    }
                    
                    # Use emotion color or default to purple
                    bg_color = colors.get(label.lower(), "#4b2996")
                    
                    # Create a color image
                    img = Image.new('RGB', (60, 60), color=bg_color)
                    
                    # Add text
                    try:
                        from PIL import ImageDraw, ImageFont
                        draw = ImageDraw.Draw(img)
                        try:
                            # Try to use a system font
                            font = ImageFont.truetype("arial.ttf", 20)
                        except:
                            font = ImageFont.load_default()
                            
                        # Draw first letter of emotion
                        letter = label[0].upper()
                        # Get text width for centering
                        try:
                            text_width = draw.textlength(letter, font=font)
                            text_height = 20  # Approximate height
                        except:
                            # Fallback if textlength not available
                            text_width = 10
                            text_height = 20
                            
                        # Position text in center
                        position = ((60 - text_width) // 2, (60 - text_height) // 2)
                        draw.text(position, letter, fill="white", font=font)
                    except Exception as e:
                        print(f"Error adding text to placeholder: {str(e)}")
                    
                    # Use the generated image
                    photo = ImageTk.PhotoImage(img)
                    self.album_cover.config(image=photo, text="")
                    self.album_cover.image = photo
                    print(f"Created placeholder album cover for {label}")
                    
                    # Try to save the image for future use if directory exists
                    if os.path.exists(images_dir):
                        try:
                            img.save(cover_path)
                            print(f"Created placeholder image: {cover_path}")
                        except Exception as e:
                            print(f"Error saving placeholder image: {str(e)}")
                except Exception as e:
                    print(f"Error creating placeholder: {str(e)}")
                    self.album_cover.config(image="", text=label[0].upper())
            else:
                # Regular image loading
                try:
                    print(f"Loading album cover from {cover_path}")
                    img = Image.open(cover_path)
                    img = img.resize((60, 60), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    self.album_cover.config(image=photo, text="")
                    self.album_cover.image = photo
                except Exception as e:
                    print(f"Error loading cover image: {str(e)}")
                    self.album_cover.config(image="", text=label[0].upper())
            
            self.play_btn.song_path = first_song
        else:
            self.song_title.config(text="No song found")
            self.artist_name.config(text="")
            self.album_cover.config(image="", text="Music")
            self.play_btn.song_path = None

    def stop_current_song(self):
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            self.currently_playing = None
            self.is_paused = False
            
            # Reset progress bar and labels
            self.progress_bar["value"] = 0
            self.current_time_label.config(text="0:00")
            self.total_time_label.config(text="0:00")
            
            # Cancel progress updates
            if self.progress_update_id:
                self.root.after_cancel(self.progress_update_id)
                self.progress_update_id = None
            
            # Stop animations
            self.stop_animation()
            self.stop_album_pulse()

    def play_song(self):
        song_path = getattr(self.play_btn, 'song_path', None)
        if song_path:
            if self.is_paused and self.currently_playing == song_path:
                # Resume if paused and it's the same song
                pygame.mixer.music.unpause()
                self.is_paused = False
                # Continue progress updates
                self.update_progress_bar()
                # Restart animation
                self.start_animation()
                # Start album cover pulsing
                self.start_album_pulse()
            else:
                # Stop any current playback
                self.stop_current_song()
                
                # Start transition animation for song change
                if self.currently_playing is not None:
                    self.start_transition_animation(song_path)
                    return
                
                # Start new song
                pygame.mixer.music.load(song_path)
                pygame.mixer.music.play()
                self.currently_playing = song_path
                
                # Get song length if possible - try multiple methods
                self.detect_song_duration(song_path)
                    
                # Reset timer variables
                self.start_time = time.time()
                self.elapsed_time = 0
                
                # Update track information display
                self.update_track_info(song_path)
                    
                # Start progress updates
                self.progress_bar["value"] = 0
                self.update_progress_bar()
                
                # Start animation
                self.start_animation()
                
                # Start album cover pulsing
                self.start_album_pulse()
        else:
            messagebox.showwarning("Warning", "No song selected.")

    def detect_song_duration(self, song_path):
        """Detect song duration using multiple methods and update display"""
        try:
            # First try with mutagen - most accurate
            try:
                import mutagen
                audio = mutagen.File(song_path)
                if audio and hasattr(audio, 'info') and hasattr(audio.info, 'length'):
                    length = audio.info.length
                    self.song_duration = length  # Store for seeking
                    mins, secs = divmod(int(length), 60)
                    self.total_time_label.config(text=f"{mins}:{secs:02d}")
                    print(f"Duration from mutagen: {mins}:{secs:02d}")
                    return
            except (ImportError, Exception) as e:
                print(f"Mutagen method failed: {e}")
                
            # If mutagen failed, try direct MP3 method for MP3 files
            if song_path.lower().endswith('.mp3'):
                try:
                    import struct
                    with open(song_path, 'rb') as f:
                        # Skip to near the end where ID3v1 tag might be
                        f.seek(-128, 2)
                        tag = f.read(3)
                        # If ID3v1 tag exists, go back further
                        if tag == b'TAG':
                            f.seek(-192, 2)
                        else:
                            # Otherwise try from the end - 64 bytes
                            f.seek(-64, 2)
                            
                        # Read chunks backwards looking for MPEG frame header
                        data = f.read(64)
                        for i in range(len(data) - 4):
                            # Check for MPEG frame header (starts with 0xFF 0xFB or similar)
                            if data[i] == 0xFF and (data[i+1] & 0xE0) == 0xE0:
                                # Found header, extract bitrate and other info
                                bitrate_index = (data[i+2] & 0xF0) >> 4
                                sampling_index = (data[i+2] & 0x0C) >> 2
                                padding = (data[i+2] & 0x02) >> 1
                                
                                # Bitrate lookup table (kbps)
                                bitrates = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
                                bitrate = bitrates[bitrate_index] * 1000
                                
                                if bitrate > 0:
                                    # Estimate duration from file size and bitrate
                                    file_size = os.path.getsize(song_path)
                                    duration_sec = file_size * 8 / bitrate
                                    self.song_duration = duration_sec
                                    mins, secs = divmod(int(duration_sec), 60)
                                    self.total_time_label.config(text=f"{mins}:{secs:02d}")
                                    print(f"Duration from MP3 header: {mins}:{secs:02d}")
                                    return
                                break
                except Exception as e:
                    print(f"MP3 header method failed: {e}")
            
            # Fallback method: estimate duration based on file size
            file_size_mb = os.path.getsize(song_path) / (1024 * 1024)
            
            # Adjust estimate based on file type
            if song_path.lower().endswith('.wav'):
                # WAV files are uncompressed, ~10MB per minute at CD quality
                estimated_seconds = file_size_mb * 6
            elif song_path.lower().endswith('.ogg'):
                # OGG files are very compressed, ~1MB per 2 minutes at decent quality
                estimated_seconds = file_size_mb * 120
            else:
                # MP3 and others, assume ~1MB per minute at 128kbps
                estimated_seconds = file_size_mb * 60
                
            # Provide a minimum duration even for small files
            self.song_duration = max(estimated_seconds, 30)  
            mins, secs = divmod(int(self.song_duration), 60)
            self.total_time_label.config(text=f"{mins}:{secs:02d}*")
            print(f"Duration estimated from size: {mins}:{secs:02d}*")
            
        except Exception as e:
            print(f"All duration detection methods failed: {e}")
            # Set a default duration value of 3 minutes for seeking to work
            self.song_duration = 180
            self.total_time_label.config(text="3:00*")
            print("Using default duration: 3:00*")
    
    def update_track_info(self, song_path):
        """Update the track information display with details about the current song"""
        if not song_path or not os.path.exists(song_path):
            self.track_info_frame.pack_forget()
            return
            
        # Make the track info frame visible
        self.track_info_frame.pack(fill=X, pady=10, padx=40, after=self.song_card)
        
        # Get basic file information
        filename = os.path.basename(song_path)
        file_size = os.path.getsize(song_path)
        size_kb = file_size / 1024
        size_mb = size_kb / 1024
        
        if size_mb >= 1:
            size_str = f"{size_mb:.2f} MB"
        else:
            size_str = f"{size_kb:.0f} KB"
        
        # Get file format
        file_ext = os.path.splitext(song_path)[1].upper().replace(".", "")
        
        # Extract title and artist
        song_title = "Unknown Title"
        artist_name = "Unknown Artist"
        
        base_filename = os.path.splitext(filename)[0]
        
        # Try to extract artist name from filename (assume format: Artist - Title)
        if " - " in base_filename:
            parts = base_filename.split(" - ", 1)
            if len(parts) == 2:
                artist_name = parts[0].title()
                song_title = parts[1].title()
        elif " by " in base_filename.lower():
            # Alternative format: Title by Artist
            parts = base_filename.lower().split(" by ", 1)
            if len(parts) == 2:
                song_title = parts[0].title()
                artist_name = parts[1].title()
        else:
            song_title = base_filename.title()
        
        # Update basic info
        self.info_title.config(text=f"Title: {song_title}")
        self.info_artist.config(text=f"Artist: {artist_name}")
        self.info_emotion.config(text=f"Emotion: {self.emotion.title() if self.emotion else 'None'}")
        
        self.info_format.config(text=f"Format: {file_ext}")
        self.info_size.config(text=f"Size: {size_str}")
        
        # Try to get duration from already detected value
        try:
            duration = "Unknown"
            if hasattr(self, 'song_duration') and self.song_duration > 0:
                mins, secs = divmod(int(self.song_duration), 60)
                duration = f"{mins}:{secs:02d}"
                # Update the total time label again to ensure it's displayed
                self.total_time_label.config(text=f"{mins}:{secs:02d}")
            self.info_duration.config(text=f"Duration: {duration}")
        except:
            self.info_duration.config(text="Duration: Unknown")
        
        # Update file path
        self.info_path.config(text=f"Path: {song_path}")
        
        # Try to get detailed audio information
        try:
            import mutagen
            audio = mutagen.File(song_path)
            
            if audio:
                # Get bitrate if available
                bitrate = "Unknown"
                if hasattr(audio.info, 'bitrate'):
                    bitrate = f"{int(audio.info.bitrate / 1000)} kbps"
                self.info_bitrate.config(text=f"Bitrate: {bitrate}")
                
                # Get sample rate if available
                sample_rate = "Unknown"
                if hasattr(audio.info, 'sample_rate'):
                    sample_rate = f"{audio.info.sample_rate} Hz"
                self.info_sample_rate.config(text=f"Sample Rate: {sample_rate}")
                
                # Get channels if available
                channels = "Unknown"
                if hasattr(audio.info, 'channels'):
                    channels_map = {1: "Mono", 2: "Stereo", 6: "5.1 Surround"}
                    channels = channels_map.get(audio.info.channels, f"{audio.info.channels} channels")
                self.info_channels.config(text=f"Channels: {channels}")
        except:
            # If mutagen fails or isn't installed
            self.info_bitrate.config(text="Bitrate: Unknown")
            self.info_sample_rate.config(text="Sample Rate: Unknown")
            self.info_channels.config(text="Channels: Unknown")

    def toggle_pause(self):
        if self.currently_playing:
            if self.is_paused:
                pygame.mixer.music.unpause()
                self.is_paused = False
                # Continue progress updates
                self.update_progress_bar()
                # Restart animation
                self.start_animation()
                # Restart album cover pulsing
                self.start_album_pulse()
            else:
                pygame.mixer.music.pause()
                self.is_paused = True
                # Stop progress updates while paused
                if self.progress_update_id:
                    self.root.after_cancel(self.progress_update_id)
                    self.progress_update_id = None
                # Pause animation
                self.stop_animation()
                # Stop album cover pulsing
                self.stop_album_pulse()

    def previous_song(self):
        if self.song_paths:
            try:
                current_index = self.song_paths.index(self.currently_playing)
            except ValueError:
                current_index = 0
            prev_index = (current_index - 1) % len(self.song_paths)
            prev_song_path = self.song_paths[prev_index]
            self.play_btn.song_path = prev_song_path
            
            # Start transition animation
            self.start_transition_animation(prev_song_path)
            return

    def next_song(self):
        if self.song_paths:
            # If in shuffle mode, pick a random song
            if self.shuffle_mode:
                next_index = random.randint(0, len(self.song_paths) - 1)
            else:
                # Regular sequential play
                try:
                    current_index = self.song_paths.index(self.currently_playing)
                except ValueError:
                    current_index = -1
                next_index = (current_index + 1) % len(self.song_paths)
                
            next_song_path = self.song_paths[next_index]
            self.play_btn.song_path = next_song_path
            
            # Start transition animation
            self.start_transition_animation(next_song_path)
            return

    def shuffle_song(self):
        """Play a random song from the playlist"""
        # First enable shuffle mode
        if not self.shuffle_mode:
            self.toggle_shuffle()
            
        # Then play a random song
        if self.song_paths:
            next_index = random.randint(0, len(self.song_paths) - 1)
            next_song_path = self.song_paths[next_index]
            self.play_btn.song_path = next_song_path
            self.stop_current_song()
            pygame.mixer.music.load(next_song_path)
            pygame.mixer.music.play()
            self.currently_playing = next_song_path
            self.song_listbox.selection_clear(0, END)
            self.song_listbox.selection_set(next_index)
            self.song_listbox.activate(next_index)
            self.song_listbox.see(next_index)
            
            # Update track information display
            self.update_track_info(next_song_path)
            
            # Reset timer for progress bar
            self.start_time = time.time()
            self.elapsed_time = 0
            self.update_progress_bar()
            
            # Start animation
            self.start_animation()

    def repeat_song(self):
        """Repeat the current song"""
        if self.currently_playing:
            # Enable repeat one mode if not already
            if not self.repeat_one:
                self.toggle_repeat()
                
            # Replay current song
            self.stop_current_song()
            
            # Important: reset progress update state to avoid conflicts
            if self.progress_update_id:
                self.root.after_cancel(self.progress_update_id)
                self.progress_update_id = None
            
            pygame.mixer.music.load(self.currently_playing)
            pygame.mixer.music.play()
            self.is_paused = False  # Ensure we're in play mode
            self.currently_playing = self.currently_playing  # Maintain reference
            
            # Reset timer for progress bar
            self.start_time = time.time()
            self.elapsed_time = 0
            
            # Show visual feedback
            self.emotion_label.config(text=f"Repeating: {os.path.basename(self.currently_playing)}")
            
            # Start the progress updates
            self.update_progress_bar()
            
            # After 2 seconds, restore the emotion label
            self.root.after(2000, lambda: self.emotion_label.config(
                text=f"Detected Emotion: {self.emotion.title() if self.emotion else 'None'}"))

    def toggle_webcam(self):
        """Toggle the webcam on/off state with improved reliability"""
        try:
            if not hasattr(self, 'webcam_active') or not self.webcam_active:
                # Starting webcam
                self.emotion_label.config(text="Starting webcam...")
                self.webcam_btn.config(text="Capture Emotion", bg='#e74c3c')  # Renamed button for clarity
                
                # Reset webcam state variables to ensure clean start
                if hasattr(self, 'cap') and self.cap is not None:
                    self.cap.release()
                    self.cap = None
                
                # Reset last detected emotion
                self.last_detected_emotion = None
                self.last_detected_confidence = 0
                self.last_emotion_frame = None
                
                self.webcam_active = True
                self.single_capture_mode = False  # Set to continuous mode
                
                # Initialize camera with optimized settings
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                
                # Set ultrafast properties - much lower resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 240)  # Reduced from 320
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)  # Reduced from 240
                self.cap.set(cv2.CAP_PROP_FPS, 10)  # Reduced from 15
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                if not self.cap.isOpened():
                    self.emotion_label.config(text="Error: Could not open webcam")
                    self.webcam_btn.config(text="Your Current Mood", bg='#4b2996')
                    self.webcam_active = False
                    return
                
                # Start updating frames
                self.update_webcam_frame()
                self.emotion_label.config(text="Webcam active - Express your emotion then press Capture")
            else:
                # Stopping webcam - will trigger analyze_last_emotion if emotion found
                self.webcam_btn.config(text="Your Current Mood", bg='#4b2996')
                self.emotion_label.config(text="Capturing current expression...")
                self.stop_webcam()
        except Exception as e:
            print(f"Error toggling webcam: {str(e)}")
            self.webcam_btn.config(text="Your Current Mood", bg='#4b2996')
            self.stop_webcam()
            messagebox.showerror("Webcam Error", f"Could not control webcam: {str(e)}")

    def stop_webcam(self):
        """Stop the webcam with improved reliability"""
        try:
            # Flag that webcam should be inactive before analysis to prevent loops
            self.webcam_active = False
            
            # Release webcam resources
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
                self.cap = None
            
            # If we have a last emotion frame, analyze it regardless of confidence
            # This ensures we capture whatever expression was showing when user clicked
            if self.last_emotion_frame is not None:
                self.analyze_last_emotion()
            else:
                # Clear the preview image if no emotion was detected
                self.preview_label.config(image='')
                self.preview_label.image = None
                self.emotion_label.config(text="No face was detected")
            
            # Reset button state
            self.webcam_btn.config(text="Your Current Mood", bg='#4b2996')
            
        except Exception as e:
            print(f"Error in stop_webcam: {str(e)}")
            # Ensure variables are reset even if error occurs
            self.webcam_active = False
            self.cap = None

    def analyze_last_emotion(self):
        """Analyze the last detected emotion from webcam"""
        # Fix the condition to properly check if there's a last emotion frame
        if not self.last_detected_emotion or self.last_emotion_frame is None:
            self.emotion_label.config(text="No emotion was detected before stopping")
            return
            
        try:
            # Update UI
            self.emotion_label.config(text=f"Analyzing last detected expression: {self.last_detected_emotion.title()}")
            self.root.update()
            
            # Display the last emotion frame with annotation
            frame = self.last_emotion_frame.copy()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Find face in the saved frame again to add rectangle
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Use faster face detection for the static image
            faces = self.face_detection.detectMultiScale(
                gray, 
                scaleFactor=1.2,
                minNeighbors=3,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Add emotion label to the image
            if len(faces) > 0:
                (x, y, w, h) = sorted(faces, reverse=True, key=lambda x: x[2] * x[3])[0]
                cv2.rectangle(rgb_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                emotion_text = f"{self.last_detected_emotion.title()} ({self.last_detected_confidence:.1f}%)"
                cv2.putText(rgb_frame, emotion_text, (x, y + h + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Convert to display image - using faster resampling
            img = Image.fromarray(rgb_frame)
            img = img.resize((350, 180), Image.Resampling.BILINEAR)  # BILINEAR is faster than LANCZOS
            photo = ImageTk.PhotoImage(image=img)
            self.preview_label.configure(image=photo)
            self.preview_label.image = photo
            
            # Store the emotion and show songs
            self.emotion = self.last_detected_emotion
            self.emotion_label.config(text=f"Emotion captured: {self.last_detected_emotion.title()} ({self.last_detected_confidence:.1f}%)")
            self.show_song_for_emotion(self.last_detected_emotion)
            
            # Auto-play first song
            if self.song_paths:
                self.play_song()
                
        except Exception as e:
            print(f"Error analyzing last emotion: {str(e)}")
            self.emotion_label.config(text=f"Error analyzing emotion: {str(e)}")

    def update_webcam_frame(self):
        """Update webcam frame with improved error handling and optimized for speed"""
        # Early exit if webcam is no longer active
        if not self.webcam_active:
            return
            
        try:
            # Check if camera is still valid
            if not hasattr(self, 'cap') or self.cap is None or not self.cap.isOpened():
                print("Webcam not accessible, stopping")
                self.stop_webcam()
                return
                
            # Read frame - just one read for speed
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                self.stop_webcam()
                return
                
            # Save a copy of the frame for potential emotion analysis
            clean_frame = frame.copy()
                
            # Ultra-fast processing - use a smaller frame for detection
            small_frame = cv2.resize(frame, (120, 90))  # Half size for detection
            
            # Convert directly to grayscale for face detection (skip RGB conversion)
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            # Simplified face detection - optimized parameters for speed
            faces = self.face_detection.detectMultiScale(
                gray, 
                scaleFactor=1.2,  # Increased from 1.1 for much faster detection
                minNeighbors=3,   # Reduced from 5 for faster detection
                minSize=(25, 25), # Reduced size requirement for scaled down image
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Convert the frame for display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # If we have faces
            if len(faces) > 0:
                # Rescale face coordinates to original frame size
                scale_x = frame.shape[1] / small_frame.shape[1]
                scale_y = frame.shape[0] / small_frame.shape[0]
                
                # Get the largest face
                (x, y, w, h) = sorted(faces, reverse=True, key=lambda x: x[2] * x[3])[0]
                
                # Scale back to original frame
                x, y = int(x * scale_x), int(y * scale_y)
                w, h = int(w * scale_x), int(h * scale_y)
                
                # Draw rectangle
                cv2.rectangle(rgb_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Process emotion only every 3rd frame to improve speed
                if hasattr(self, 'frame_count'):
                    self.frame_count += 1
                else:
                    self.frame_count = 0
                
                # Skip emotion processing on some frames for speed
                if self.frame_count % 3 == 0:
                    # Process emotion on original frame
                    roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y + h, x:x + w]
                    
                    # Ensure ROI is valid
                    if roi_gray.size > 0 and roi_gray.shape[0] > 20 and roi_gray.shape[1] > 20:
                        # Direct resize to 48x48 for emotion detection
                        roi = cv2.resize(roi_gray, (48, 48))
                        
                        # Streamlined preprocessing
                        roi = roi.astype("float") / 255.0
                        roi = img_to_array(roi)
                        roi = np.expand_dims(roi, axis=0)
        
                        # Predict emotion
                        preds = self.emotion_classifier.predict(roi)[0]
                        confidence = preds.max() * 100
                        detected_emotion = self.EMOTIONS[preds.argmax()]
                        
                        # Store last detected emotion and frame
                        self.last_detected_emotion = detected_emotion
                        self.last_detected_confidence = confidence
                        self.last_emotion_frame = clean_frame
                        
                        # Display emotion label on image
                        emotion_text = f"{detected_emotion.title()} ({confidence:.1f}%)"
                        cv2.putText(rgb_frame, emotion_text, (x, y + h + 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                        
                        # Update UI with emotion - only update text if emotion has changed
                        if not hasattr(self, 'last_displayed_emotion') or self.last_displayed_emotion != detected_emotion:
                            self.emotion_label.config(text=f"Live: {detected_emotion.title()}")
                            self.last_displayed_emotion = detected_emotion
            
            # Fast image conversion and display
            img = Image.fromarray(rgb_frame)
            img = img.resize((350, 180), Image.Resampling.NEAREST)  # Using NEAREST for speed instead of LANCZOS
            photo = ImageTk.PhotoImage(image=img)
            
            # Update preview
            self.preview_label.configure(image=photo)
            self.preview_label.image = photo
            
            # Schedule next update with decreased frequency for better performance
            if self.webcam_active:
                self.root.after(33, self.update_webcam_frame)  # Decreased refresh frequency
                
        except Exception as e:
            print(f"Error in update_webcam_frame: {str(e)}")
            # On error, try to safely stop the webcam
            self.stop_webcam()
            
    def analyze_with_webcam(self):
        """Capture emotion from webcam with one click - ultra fast version"""
        # Make sure any previous webcam session is closed
        self.stop_webcam()
        
        # Reset state
        self.webcam_active = False
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
            
        self.emotion_label.config(text="Initializing camera...")
        self.root.update()
        
        try:
            # Initialize webcam with ultra-fast settings
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 240)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not self.cap.isOpened():
                self.stop_detection_animation()
                messagebox.showerror("Error", "Could not access webcam")
                return
                
            # Quick initialization - just one frame
            self.cap.read()  # Discard first frame
            
            # Quick face detection
            self.emotion_label.config(text="Looking for face...")
            self.root.update()
            
            # Fast face detection approach
            found_face = False
            best_frame = None
            max_attempts = 5  # Reduced from 10
            
            for attempt in range(max_attempts):
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Show preview
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                img = img.resize((350, 180), Image.Resampling.NEAREST)
                photo = ImageTk.PhotoImage(image=img)
                self.preview_label.configure(image=photo)
                self.preview_label.image = photo
                
                # Fast face detection
                small_frame = cv2.resize(frame, (120, 90))
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_detection.detectMultiScale(
                    gray, scaleFactor=1.2, minNeighbors=3, minSize=(25, 25)
                )
                
                if len(faces) > 0:
                    found_face = True
                    best_frame = frame.copy()
                    self.emotion_label.config(text="Face found! Capturing emotion...")
                    break
                
                self.emotion_label.config(text=f"Looking for face... ({attempt+1}/{max_attempts})")
                self.root.update()
            
            # If no face found, stop
            if not found_face or best_frame is None:
                self.stop_detection_animation()
                self.emotion_label.config(text="No face detected")
                self.cap.release()
                self.cap = None
                return
            
            # Quick capture of emotion
            rgb_frame = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(best_frame, cv2.COLOR_BGR2GRAY)
            
            # Get face region
            faces = self.face_detection.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=3, minSize=(30, 30)
            )
            
            if len(faces) > 0:
                (x, y, w, h) = sorted(faces, reverse=True, key=lambda x: x[2] * x[3])[0]
                roi = gray[y:y+h, x:x+w]
                
                if roi.size > 0 and roi.shape[0] > 20 and roi.shape[1] > 20:
                    # Process emotion
                    roi = cv2.resize(roi, (48, 48))
                    roi = roi.astype("float") / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)
                    
                    # Get emotion
                    preds = self.emotion_classifier.predict(roi)[0]
                    confidence = preds.max() * 100
                    emotion = self.EMOTIONS[preds.argmax()]
                    
                    # Stop the detection animation
                    self.stop_detection_animation()
                    
                    # Draw on image
                    cv2.rectangle(rgb_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    emotion_text = f"{emotion.title()} ({confidence:.1f}%)"
                    cv2.putText(rgb_frame, emotion_text, (x, y + h + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    # Show result
                    img = Image.fromarray(rgb_frame)
                    img = img.resize((350, 180), Image.Resampling.BILINEAR)
                    photo = ImageTk.PhotoImage(image=img)
                    self.preview_label.configure(image=photo)
                    self.preview_label.image = photo
                    
                    # Update UI
                    self.emotion = emotion
                    self.emotion_label.config(text=f"Detected: {emotion.title()} ({confidence:.1f}%)")
                    
                    # Recommend songs
                    self.show_song_for_emotion(emotion)
                    
                    # Play first song
                    if self.song_paths:
                        self.play_song()
                else:
                    self.stop_detection_animation()
                    self.emotion_label.config(text="Face detected but too small for analysis")
            else:
                self.stop_detection_animation()
                self.emotion_label.config(text="Lost face during analysis")
            
            # Clean up
            self.cap.release()
            self.cap = None
            
        except Exception as e:
            self.stop_detection_animation()
            print(f"Error: {str(e)}")
            self.emotion_label.config(text=f"Error: {str(e)}")
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
                self.cap = None

    def on_song_select(self, event):
        """Play song immediately when clicked in the listbox"""
        try:
            selection = event.widget.curselection()
            if selection:
                index = selection[0]
                song_path = self.song_paths[index]
                self.play_btn.song_path = song_path  # Update play button to play this song
                
                # Start the transition animation
                self.start_transition_animation(song_path)
        except Exception as e:
            print(f"Error playing song: {str(e)}")
            messagebox.showerror("Playback Error", f"Could not play song: {str(e)}")

    def on_close(self):
        # Cancel any pending progress updates
        if self.progress_update_id:
            self.root.after_cancel(self.progress_update_id)
        # Cancel any animation frames
        if self.animation_frame_id:
            self.root.after_cancel(self.animation_frame_id)
        if self.album_pulse_id:
            self.root.after_cancel(self.album_pulse_id)
        if self.transition_id:
            self.root.after_cancel(self.transition_id)
        if hasattr(self, 'detection_animation_id') and self.detection_animation_id:
            self.root.after_cancel(self.detection_animation_id)
        self.stop_current_song()
        self.root.destroy()

    def show_emotion_selector(self):
        """Show a dialog with emotion options for text-based selection"""
        # Create a new toplevel window
        selector = Toplevel(self.root)
        selector.title("Select Your Emotion")
        selector.geometry("300x350")
        selector.configure(bg='#f6f6fa')
        selector.resizable(False, False)
        
        # Make it modal
        selector.transient(self.root)
        selector.grab_set()
        
        # Add a title
        Label(selector, text="How are you feeling?", 
              font=('Segoe UI', 16, 'bold'), fg='#4b2996', bg='#f6f6fa').pack(pady=(20, 25))
        
        # Create a frame for emotion buttons
        btn_frame = Frame(selector, bg='#f6f6fa')
        btn_frame.pack(fill=BOTH, expand=True, padx=30, pady=10)
        
        # Define emotion colors
        emotion_colors = {
            "happy": "#ffd166",    # Yellow
            "sad": "#118ab2",      # Blue
            "angry": "#ef476f",    # Red
            "scared": "#7209b7",   # Purple
            "surprised": "#06d6a0", # Teal
            "neutral": "#8d99ae",  # Gray-Blue
            "disgust": "#6a994e"   # Green
        }
        
        # Function to handle button click
        def select_emotion(emotion):
            self.on_text_emotion_selected(emotion)
            selector.destroy()
        
        # Create a button for each emotion with appropriate styling
        row, col = 0, 0
        for emotion in self.EMOTIONS:
            bg_color = emotion_colors.get(emotion, "#4b2996")
            fg_color = "#fff" if emotion != "happy" else "#333"
            
            # Create button with specific styling
            btn = Button(
                btn_frame, 
                text=emotion.title(), 
                font=('Segoe UI', 12, 'bold'),
                bg=bg_color, 
                fg=fg_color,
                bd=0,
                width=10,
                height=2,
                cursor="hand2",
                command=lambda e=emotion: select_emotion(e)
            )
            
            # Place in grid
            btn.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")
            
            # Update grid position
            col += 1
            if col > 1:  # 2 columns
                col = 0
                row += 1
                
        # Make grid cells expand properly
        for i in range(4):  # 4 rows
            btn_frame.grid_rowconfigure(i, weight=1)
        for i in range(2):  # 2 columns
            btn_frame.grid_columnconfigure(i, weight=1)
            
        # Add a cancel button
        Button(selector, text="Cancel", font=('Segoe UI', 11),
               command=selector.destroy, bg='#f0f0f8', fg='#666',
               bd=0, padx=15, pady=5).pack(pady=(5, 20))
        
        # Center the window on screen
        selector.update_idletasks()
        width = selector.winfo_width()
        height = selector.winfo_height()
        x = (selector.winfo_screenwidth() // 2) - (width // 2)
        y = (selector.winfo_screenheight() // 2) - (height // 2)
        selector.geometry(f'{width}x{height}+{x}+{y}')
    
    def on_text_emotion_selected(self, emotion):
        """Handle when an emotion is selected via text"""
        try:
            # Update the emotion label
            self.emotion_label.config(text=f"Selected Emotion: {emotion.title()}")
            
            # Store the emotion
            self.emotion = emotion
            self.last_detected_emotion = emotion
            self.last_detected_confidence = 100  # Text selection has perfect confidence
            
            # Clear any image in the preview
            self.preview_label.config(image='')
            
            # Create a colorful display showing the selected emotion
            colors = {
                "happy": "#ffd166",    # Yellow
                "sad": "#118ab2",      # Blue
                "angry": "#ef476f",    # Red
                "scared": "#7209b7",   # Purple
                "surprised": "#06d6a0", # Teal
                "neutral": "#8d99ae",  # Gray-Blue
                "disgust": "#6a994e"   # Green
            }
            
            # Create a blank image with the emotion color
            bg_color = colors.get(emotion, "#4b2996")
            img = Image.new('RGB', (350, 180), color=bg_color)
            
            # Add text to the image
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            
            # Try to get a nice font, or fall back to default
            try:
                font_path = "arial.ttf"  # Standard font on Windows
                font = ImageFont.truetype(font_path, 36)
            except:
                font = ImageFont.load_default()
                
            # Draw emotion text in the center
            text = emotion.upper()
            text_width = draw.textlength(text, font=font)
            position = ((350 - text_width) // 2, 65)
            draw.text(position, text, fill="white", font=font)
            
            # Convert to Tkinter PhotoImage
            photo = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=photo)
            self.preview_label.image = photo
            
            # Recommend songs based on the emotion
            self.show_song_for_emotion(emotion)
            
            # Optionally auto-play the first song
            if self.song_paths:
                self.play_song()
                
        except Exception as e:
            print(f"Error handling text emotion: {str(e)}")
            messagebox.showerror("Error", f"Error selecting emotion: {str(e)}")

    def activate_language_menu(self):
        """Activate the language selection dropdown once an emotion is detected"""
        self.language_menu.config(state=NORMAL)
        self.language_label.config(fg='#4b2996')  # Change text color to indicate active
        
        # Add visual feedback to show the menu is now active
        self.language_label.config(text="Select Language:")
        
    def on_language_change(self):
        """Handle language change by refreshing songs for the current emotion"""
        if self.emotion and self.emotion != "None":
            # Show the loading status
            self.emotion_label.config(text=f"Loading {self.language_var.get()} songs for {self.emotion}...")
            self.root.update()
            
            # Refresh songs for the current emotion with the new language
            self.show_song_for_emotion(self.emotion)
            
            # If we had a song playing, stop it as the language changed
            self.stop_current_song()
            
            # Play the first song in the new language if available
            if self.song_paths:
                self.play_song()
                
            # Update status - remove the language changed message
            # Just keep the previous emotion message instead of showing language changed
            self.emotion_label.config(text=f"Detected Emotion: {self.emotion.title()}")

    def toggle_mute(self):
        """Toggle mute/unmute audio"""
        if self.is_muted:
            # Unmute
            pygame.mixer.music.set_volume(self.previous_volume)
            self.current_volume = self.previous_volume
            self.volume_var.set(self.current_volume * 100)
            self.volume_value_label.config(text=f"{int(self.current_volume*100)}%")
            self.mute_btn.config(text="Mute", bg='#4b2996')
            self.volume_icon_label.config(text="Vol:")
            self.is_muted = False
        else:
            # Mute
            self.previous_volume = self.current_volume
            pygame.mixer.music.set_volume(0)
            self.volume_var.set(0)
            self.volume_value_label.config(text="0%")
            self.mute_btn.config(text="Unmute", bg='#e74c3c')
            self.volume_icon_label.config(text="Muted:")
            self.is_muted = True

    def set_volume(self, val):
        volume = float(val) / 100.0
        pygame.mixer.music.set_volume(volume)
        self.current_volume = volume
        self.volume_value_label.config(text=f"{int(float(val))}%")
        
        # Update volume icon based on level
        if volume == 0:
            self.volume_icon_label.config(text="Muted:")
        elif volume < 0.3:
            self.volume_icon_label.config(text="Low:")
        elif volume < 0.7:
            self.volume_icon_label.config(text="Med:")
        else:
            self.volume_icon_label.config(text="Vol:")
        
        # Update mute button if volume is changed
        if volume > 0 and self.is_muted:
            self.is_muted = False
            self.mute_btn.config(text="Mute")
        elif volume == 0 and not self.is_muted:
            self.is_muted = True
            self.mute_btn.config(text="Unmute")

    def update_progress_bar(self):
        """Update the progress bar based on current playback position"""
        if pygame.mixer.music.get_busy() and not self.is_paused:
            # Get current position (pygame doesn't provide this directly, so we track time)
            if not hasattr(self, 'start_time'):
                self.start_time = time.time()
                self.elapsed_time = 0
            else:
                self.elapsed_time = time.time() - self.start_time
            
            # Update progress bar
            mins, secs = divmod(int(self.elapsed_time), 60)
            self.current_time_label.config(text=f"{mins}:{secs:02d}")
            
            # Calculate a percentage for the progress bar (rough estimate)
            song_finished = False
            try:
                total_text = self.total_time_label.cget("text")
                if total_text != "--:--":
                    # Safely extract minutes and seconds, handling any extra characters
                    min_sec = total_text.split(":")
                    if len(min_sec) >= 2:
                        # Extract only the numeric part of seconds (ignore any suffix like "*")
                        minutes = int(min_sec[0])
                        seconds_part = min_sec[1].strip()
                        # Extract only digits from the seconds part
                        seconds = int(''.join(c for c in seconds_part if c.isdigit()))
                        
                        total_secs = minutes * 60 + seconds
                        if total_secs > 0:
                            percentage = (self.elapsed_time / total_secs) * 100
                            self.progress_bar["value"] = min(percentage, 100)
                            
                            # Check if song has reached end threshold
                            if percentage > (self.end_of_song_threshold * 100):
                                song_finished = True
            except Exception as e:
                print(f"Error updating progress: {e}")
            
            # Handle song end detection
            if song_finished and hasattr(self, 'song_duration') and self.elapsed_time >= (self.song_duration * self.end_of_song_threshold):
                print(f"Song end detected at {self.elapsed_time:.2f}s of {self.song_duration:.2f}s")
                # Cancel this update cycle
                if self.progress_update_id:
                    self.root.after_cancel(self.progress_update_id)
                    self.progress_update_id = None
                
                # Handle repeat or next song
                if self.repeat_one:
                    print("Repeating current song - end reached")
                    self.root.after(100, self.repeat_song)  # Small delay to ensure clean playback
                else:
                    print("Playing next song - end reached")
                    self.root.after(100, self.next_song)  # Small delay to ensure clean playback
                return
            
            # Schedule next update
            self.progress_update_id = self.root.after(1000, self.update_progress_bar)
        else:
            # If song finished (not paused but no longer busy)
            if not self.is_paused and not pygame.mixer.music.get_busy() and self.currently_playing:
                print("Song finished by pygame event")
                if self.repeat_one:
                    # Repeat the current song
                    print("Repeating current song - pygame event")
                    self.repeat_song()
                else:
                    # Play next song automatically
                    print("Playing next song - pygame event")
                    self.next_song()
            elif self.is_paused:
                # Song is paused, don't schedule updates
                pass
            else:
                # Reset progress bar
                self.progress_bar["value"] = 0
                self.current_time_label.config(text="0:00")

    def seek(self, event):
        """Handle seeking when the progress bar is clicked"""
        if not self.currently_playing:
            return
            
        # Set a fallback duration if song_duration is not set
        if not hasattr(self, 'song_duration') or self.song_duration <= 0:
            # Default to 3 minutes if we can't determine length
            self.song_duration = 180
            
        # Calculate the seek position based on where the user clicked
        widget_width = event.widget.winfo_width()
        click_position = event.x / widget_width
        
        if click_position < 0:
            click_position = 0
        elif click_position > 1:
            click_position = 1
            
        # Reset progress
        if self.progress_update_id:
            self.root.after_cancel(self.progress_update_id)
            
        try:
            # Calculate the time to seek to
            seek_time = click_position * self.song_duration
            
            # Pygame doesn't support direct seeking, so we need to reload and play from position
            pygame.mixer.music.stop()
            pygame.mixer.music.load(self.currently_playing)
            
            # Safety check for seek_time
            if seek_time >= self.song_duration:
                seek_time = max(0, self.song_duration - 1)
                
            # Try to play from the seek position
            try:
                pygame.mixer.music.play(start=seek_time)
            except Exception as e:
                print(f"Error with seek using start parameter: {e}")
                # Fallback: reload and play without seeking - better than crashing
                pygame.mixer.music.play()
                
            # Update display
            self.start_time = time.time() - seek_time
            self.elapsed_time = seek_time
            
            mins, secs = divmod(int(seek_time), 60)
            self.current_time_label.config(text=f"{mins}:{secs:02d}")
            
            # Update progress bar directly
            if self.song_duration > 0:
                percentage = (seek_time / self.song_duration) * 100
                self.progress_bar["value"] = percentage
            
            # If it was paused, pause it again
            if self.is_paused:
                pygame.mixer.music.pause()
            else:
                # Otherwise restart progress updates
                self.update_progress_bar()
                
        except Exception as e:
            print(f"Error seeking: {e}")
            # Try to recover by just playing from the start
            pygame.mixer.music.stop()
            pygame.mixer.music.load(self.currently_playing)
            pygame.mixer.music.play()
            
            # Reset timer
            self.start_time = time.time()
            self.elapsed_time = 0
            self.update_progress_bar()

    def toggle_track_details(self):
        """Toggle the display of detailed track information"""
        if self.show_details_var.get():
            self.detailed_info_frame.pack(fill=X, pady=(10, 0))
        else:
            self.detailed_info_frame.pack_forget()

    def increase_volume(self, *args):
        """Increase volume by 5%"""
        current = self.volume_var.get()
        new_volume = min(current + 5, 100)
        self.volume_var.set(new_volume)
        self.set_volume(new_volume)
        
    def decrease_volume(self, *args):
        """Decrease volume by 5%"""
        current = self.volume_var.get()
        new_volume = max(current - 5, 0)
        self.volume_var.set(new_volume)
        self.set_volume(new_volume)

    def toggle_shuffle(self):
        """Toggle shuffle mode on/off"""
        self.shuffle_mode = not self.shuffle_mode
        
        # Update button appearance based on state
        if self.shuffle_mode:
            self.shuffle_btn.config(bg='#2ecc71', text="Shuffle On")
        else:
            self.shuffle_btn.config(bg='#4b2996', text="Shuffle")
            
        # Show status in emotion label temporarily
        current_text = self.emotion_label.cget("text")
        self.emotion_label.config(text=f"Shuffle mode {'ON' if self.shuffle_mode else 'OFF'}")
        
        # Reset label after 2 seconds
        self.root.after(2000, lambda: self.emotion_label.config(text=current_text))

    def toggle_repeat(self):
        """Toggle repeat mode between: off -> repeat one -> off"""
        # Toggle repeat one mode
        self.repeat_one = not self.repeat_one
        
        # Update button appearance based on state
        if self.repeat_one:
            self.repeat_btn.config(bg='#2ecc71', text="Repeat One")
            # Make the repeat button flash once to confirm activation
            self.repeat_btn.config(bg='#e74c3c')  # Flash red
            self.root.after(100, lambda: self.repeat_btn.config(bg='#2ecc71'))  # Return to green
        else:
            self.repeat_btn.config(bg='#4b2996', text="Repeat")
            
        # Show status in emotion label with more detail
        current_text = self.emotion_label.cget("text")
        if self.repeat_one:
            self.emotion_label.config(text="Repeat ONE: Current song will repeat")
        else:
            self.emotion_label.config(text="Repeat OFF: Normal playback")
        
        # Reset label after 2 seconds
        self.root.after(2000, lambda: self.emotion_label.config(text=current_text))
        
        print(f"Repeat mode toggled: {self.repeat_one}")
        
        # Update current song state if playing
        if self.currently_playing and pygame.mixer.music.get_busy():
            song_name = os.path.basename(self.currently_playing)
            if self.repeat_one:
                print(f"Song '{song_name}' will repeat when finished")
            else:
                print(f"Song '{song_name}' will not repeat when finished")

    def create_equalizer_bars(self):
        """Create the equalizer bars on the animation canvas"""
        # Clear any existing bars
        self.animation_canvas.delete("all")
        self.equalizer_bars = []
        
        # Number of bars and their properties
        num_bars = 5  # Increased from 4 to 5 bars
        bar_width = 5
        bar_spacing = 3
        bar_color = "#4b2996"  # Same color as the app's theme
        
        # Initialize heights and directions for each bar
        self.bar_heights = [15, 22, 28, 20, 16]  # Starting heights
        self.bar_directions = [1, -1, 1, -1, 1]  # 1 = up, -1 = down
        self.base_heights = [15, 22, 28, 20, 16]  # Base heights to return to
        
        # Create the bars
        for i in range(num_bars):
            x1 = i * (bar_width + bar_spacing) + 4
            y1 = 60 - self.bar_heights[i]  # Bar grows from bottom to top
            x2 = x1 + bar_width
            y2 = 60
            
            # Create a gradient fill color based on position
            bar = self.animation_canvas.create_rectangle(
                x1, y1, x2, y2, 
                fill=bar_color, 
                outline=""  # No outline for cleaner look
            )
            self.equalizer_bars.append(bar)
        
        # Add music note icon in background
        try:
            note_x = 20  # Center position
            note_y = 30
            # Create a simple music note using lines and ovals
            # Note head
            self.animation_canvas.create_oval(
                note_x-7, note_y-6, note_x+1, note_y+2, 
                fill="#e0e0e6", outline="", tags="note"
            )
            # Note stem
            self.animation_canvas.create_line(
                note_x, note_y-4, note_x, note_y-20, 
                fill="#e0e0e6", width=2, tags="note"
            )
            # Note flag
            self.animation_canvas.create_arc(
                note_x, note_y-25, note_x+10, note_y-15, 
                start=0, extent=180, style=ARC,
                outline="#e0e0e6", width=2, tags="note"
            )
        except Exception as e:
            print(f"Error creating music note: {e}")
    
    def animate_equalizer(self):
        """Animate the equalizer bars with a bouncing effect and volume sensitivity"""
        if not self.animation_active:
            return
        
        # Get current song volume to affect animation
        current_vol = self.current_volume
        
        # Apply volume-based multiplier to animation amplitude
        if current_vol > 0:
            volume_factor = 1.0 + current_vol * 1.5  # Scales from 1.0 to 2.5 based on volume
        else:
            volume_factor = 0.5  # Minimal movement when volume is 0
            
        # Update each bar with a new height
        for i, bar in enumerate(self.equalizer_bars):
            # Add randomness to create more natural movement
            random_factor = random.uniform(0.8, 1.2)
            
            # Update height based on direction, volume and randomness
            self.bar_heights[i] += self.bar_directions[i] * 2 * volume_factor * random_factor
            
            # Reverse direction if reaching limits
            max_height = 45 * volume_factor  # Volume affects maximum height
            min_height = self.base_heights[i] * 0.7  # Never go below base height * factor
            
            if self.bar_heights[i] >= max_height:
                self.bar_directions[i] = -1
            elif self.bar_heights[i] <= min_height:
                self.bar_directions[i] = 1
                
            # Redraw the bar with color based on height (taller = brighter)
            x1, _, x2, _ = self.animation_canvas.coords(bar)
            y1 = 60 - self.bar_heights[i]
            y2 = 60
            
            # Calculate color brightness based on height and volume
            brightness = int(180 + min(self.bar_heights[i], 40) * 1.5)  # Varies from 180-240
            
            # Create color in rgb
            r = min(75 + brightness // 3, 255)  # More red for louder
            g = min(41 + brightness // 5, 255)  
            b = min(150 + brightness // 2, 255)  # More violet/purple base
            
            # Convert to hex for tkinter
            color = f'#{r:02x}{g:02x}{b:02x}'
            
            # Update bar position and color
            self.animation_canvas.coords(bar, x1, y1, x2, y2)
            self.animation_canvas.itemconfig(bar, fill=color)
            
        # Animate the music note opacity for additional effect
        try:
            # Pulse the music note with volume
            note_opacity = int(150 + 105 * current_vol)  # 150-255 based on volume
            note_color = f'#{note_opacity:02x}{note_opacity:02x}{note_opacity:02x}'
            self.animation_canvas.itemconfig("note", fill=note_color)
        except Exception:
            pass
        
        # Make animation smoother when volume is higher
        if current_vol > 0.5:
            # Higher volume = faster animation
            delay = max(80, 150 - int(current_vol * 80))  # 110-70ms delay
        else:
            # Lower volume = slower animation
            delay = 150  # Default 150ms delay
        
        # Schedule the next animation frame
        self.animation_frame_id = self.root.after(delay, self.animate_equalizer)

    def start_animation(self):
        """Start the music visualization animation"""
        if not self.animation_active:
            self.animation_active = True
            
            # Make the animation canvas more visible when music is playing
            self.animation_canvas.config(bg="#f0f0f8")
            
            # Randomize starting positions slightly
            for i in range(len(self.bar_heights)):
                self.bar_heights[i] = self.base_heights[i] + random.randint(-3, 3)
            
            # Begin animation
            self.animate_equalizer()
    
    def stop_animation(self):
        """Stop the music visualization animation"""
        self.animation_active = False
        if self.animation_frame_id:
            self.root.after_cancel(self.animation_frame_id)
            self.animation_frame_id = None
        
        # Return background to normal
        self.animation_canvas.config(bg="#f8f7fc")
        
        # Reset bars to inactive position
        for i, bar in enumerate(self.equalizer_bars):
            x1, _, x2, _ = self.animation_canvas.coords(bar)
            # Set all bars to same height when inactive
            y1 = 45
            y2 = 60
            self.animation_canvas.coords(bar, x1, y1, x2, y2)
            # Reset colors to default
            self.animation_canvas.itemconfig(bar, fill="#4b2996")
            
        # Fade the music note
        try:
            self.animation_canvas.itemconfig("note", fill="#e0e0e6")
        except Exception:
            pass

    # Add new album cover pulsing animation
    def start_album_pulse(self):
        """Start pulsing animation for album cover"""
        if not self.album_pulse_active:
            self.album_pulse_active = True
            self.album_scale = 1.0
            self.album_scale_direction = 0.005
            self.animate_album_pulse()
    
    def stop_album_pulse(self):
        """Stop pulsing animation for album cover"""
        self.album_pulse_active = False
        if self.album_pulse_id:
            self.root.after_cancel(self.album_pulse_id)
            self.album_pulse_id = None
        
        # Reset the album cover to normal size
        if hasattr(self, 'album_cover'):
            self.album_cover.config(font=('Segoe UI', 24))
            
    def animate_album_pulse(self):
        """Animate the album cover with a subtle pulsing effect"""
        if not self.album_pulse_active:
            return
        
        # Update the scale factor
        self.album_scale += self.album_scale_direction
        
        # Reverse direction at limits
        if self.album_scale >= 1.05:  # Max 5% larger
            self.album_scale = 1.05
            self.album_scale_direction = -0.005
        elif self.album_scale <= 0.95:  # Min 5% smaller
            self.album_scale = 0.95
            self.album_scale_direction = 0.005
            
        # Apply scaling to album cover if it has an image
        if hasattr(self, 'album_cover') and hasattr(self.album_cover, 'image') and self.album_cover.image:
            # Scale the font size for text-based album covers
            font_size = int(24 * self.album_scale)
            self.album_cover.config(font=('Segoe UI', font_size))
            
            # For actual images, we'd need to scale the image itself
            # This requires more complex handling as we'd need to rescale the original image
            # That part would be implemented for a more complete solution
        
        # Schedule next animation frame
        self.album_pulse_id = self.root.after(50, self.animate_album_pulse)
    
    # Add song transition animation
    def start_transition_animation(self, next_song_path):
        """Start a transition animation when changing songs"""
        self.transition_active = True
        self.transition_alpha = 0
        self.next_song_path = next_song_path
        
        # Create a semi-transparent overlay
        self.transition_overlay = Frame(self.root, bg='#000000', width=900, height=800)
        self.transition_overlay.place(x=0, y=0, relwidth=1, relheight=1)
        self.transition_overlay.configure(alpha=0)  # Start fully transparent
        
        # Start the animation
        self.animate_transition()
    
    def animate_transition(self):
        """Animate the transition between songs"""
        if not self.transition_active:
            return
            
        # Update alpha value
        self.transition_alpha += 0.1
        
        if self.transition_alpha <= 1.0:
            # Fading out phase
            alpha = min(self.transition_alpha, 1.0)
            # Apply transparency to overlay (if supported)
            try:
                self.transition_overlay.configure(alpha=alpha)
            except:
                # If alpha not supported, use bg color darkness as fallback
                darkness = int(255 * (1 - alpha))
                bg_color = f'#{darkness:02x}{darkness:02x}{darkness:02x}'
                self.transition_overlay.configure(bg=bg_color)
                
            self.transition_id = self.root.after(50, self.animate_transition)
        else:
            # When fully faded out, switch the song
            self.stop_current_song()
            
            # Load and play the next song
            pygame.mixer.music.load(self.next_song_path)
            pygame.mixer.music.play()
            self.currently_playing = self.next_song_path
            
            # Update song info and listbox selection
            try:
                # Find index and update selection
                next_index = self.song_paths.index(self.next_song_path)
                self.song_listbox.selection_clear(0, END)
                self.song_listbox.selection_set(next_index)
                self.song_listbox.activate(next_index)
                self.song_listbox.see(next_index)
            except ValueError:
                pass
                
            # Update track information
            self.update_track_info(self.next_song_path)
            
            # Reset timer for progress
            self.start_time = time.time()
            self.elapsed_time = 0
            self.update_progress_bar()
            
            # Start animations
            self.start_animation()
            self.start_album_pulse()
            
            # Start fade in animation
            self.start_fade_in()
    
    def start_fade_in(self):
        """Start fade in animation after song change"""
        self.transition_alpha = 1.0
        self.fade_in()
    
    def fade_in(self):
        """Animate fading back in after song change"""
        # Decrease alpha (fade in)
        self.transition_alpha -= 0.1
        
        if self.transition_alpha >= 0:
            # Apply transparency
            alpha = max(self.transition_alpha, 0)
            try:
                self.transition_overlay.configure(alpha=alpha)
            except:
                # Fallback for alpha not supported
                brightness = int(255 * alpha)
                bg_color = f'#{brightness:02x}{brightness:02x}{brightness:02x}'
                self.transition_overlay.configure(bg=bg_color)
                
            self.transition_id = self.root.after(50, self.fade_in)
        else:
            # Animation complete, clean up
            self.transition_overlay.destroy()
            self.transition_active = False
            self.transition_id = None

    # Add button hover effect function
    def add_button_hover_effect(self, button, hover_bg=None, hover_fg=None, original_bg=None, original_fg=None):
        """Add hover effect to a button"""
        if not original_bg:
            original_bg = button['bg']
        if not original_fg:
            original_fg = button['fg']
        
        # Use provided hover colors or generate them
        if not hover_bg:
            # Convert color names to hex if needed
            try:
                # Handle named colors by creating a temporary label to get the hex value
                temp = Label(self.root, bg=original_bg)
                actual_bg = temp.cget('bg')
                temp.destroy()
                
                # Now we can safely parse the hex color
                if actual_bg.startswith('#') and len(actual_bg) == 7:
                    # Standard #RRGGBB format
                    r = int(actual_bg[1:3], 16)
                    g = int(actual_bg[3:5], 16)
                    b = int(actual_bg[5:7], 16)
                    
                    # Make color lighter
                    r = min(255, r + 30)
                    g = min(255, g + 30)
                    b = min(255, b + 30)
                    
                    hover_bg = f'#{r:02x}{g:02x}{b:02x}'
                else:
                    # Fallback for unsupported colors - use a safe light purple
                    hover_bg = '#7c3aed'
            except Exception:
                # If any error occurs, use a safe color
                hover_bg = '#7c3aed'
                
        if not hover_fg:
            hover_fg = original_fg
            
        # Store original colors as button attributes
        button.original_bg = original_bg
        button.original_fg = original_fg
        button.hover_bg = hover_bg
        button.hover_fg = hover_fg
            
        # Define hover enter and leave functions
        def on_enter(e):
            e.widget['bg'] = e.widget.hover_bg
            e.widget['fg'] = e.widget.hover_fg
            
            # Add subtle animation - slightly increase font size
            try:
                current_font = e.widget['font']
                if isinstance(current_font, str):
                    font_parts = current_font.split()
                    if len(font_parts) >= 2:
                        # Try to extract size and increase it
                        try:
                            size = int(font_parts[-1])
                            new_size = size + 1
                            new_font = " ".join(font_parts[:-1] + [str(new_size)])
                            e.widget['font'] = new_font
                        except ValueError:
                            pass
            except Exception:
                pass  # Ignore font errors
            
        def on_leave(e):
            e.widget['bg'] = e.widget.original_bg
            e.widget['fg'] = e.widget.original_fg
            
            # Restore original font size
            try:
                current_font = e.widget['font']
                if isinstance(current_font, str):
                    font_parts = current_font.split()
                    if len(font_parts) >= 2:
                        # Try to extract size and decrease it back
                        try:
                            size = int(font_parts[-1])
                            new_size = size - 1
                            new_font = " ".join(font_parts[:-1] + [str(new_size)])
                            e.widget['font'] = new_font
                        except ValueError:
                            pass
            except Exception:
                pass  # Ignore font errors
        
        # Bind the hover events
        button.bind("<Enter>", on_enter)
        button.bind("<Leave>", on_leave)

    # Add emotion detection animation
    def start_detection_animation(self):
        """Start an animation to show that emotion detection is in progress"""
        if hasattr(self, 'detection_animation_active') and self.detection_animation_active:
            return
            
        self.detection_animation_active = True
        
        # Create a canvas overlay on the preview frame for the animation
        self.detection_canvas = Canvas(self.preview_frame, width=350, height=180, 
                                      bg='#222', highlightthickness=0)
        self.detection_canvas.place(x=0, y=0, relwidth=1, relheight=1)
        
        # Create dots for the loading animation
        self.detection_dots = []
        num_dots = 8
        center_x, center_y = 175, 90  # Center of the preview frame
        radius = 40  # Radius of the circle the dots will rotate in
        
        # Create dots spaced evenly around a circle
        for i in range(num_dots):
            angle = i * (360 / num_dots)
            x = center_x + radius * np.cos(np.radians(angle))
            y = center_y + radius * np.sin(np.radians(angle))
            
            # Size decreases with distance from start
            size = 8 - (i % 5)
            if size < 3:
                size = 3
                
            # Color gets lighter with distance from start
            brightness = 155 + (i * 10)
            if brightness > 255:
                brightness = 255
                
            color = f'#{brightness:02x}{brightness//2:02x}{brightness:02x}'
            
            dot = self.detection_canvas.create_oval(
                x-size, y-size, x+size, y+size,
                fill=color, outline=""
            )
            self.detection_dots.append(dot)
        
        # Add processing text
        self.detection_text = self.detection_canvas.create_text(
            center_x, center_y,
            text="Processing...",
            font=('Segoe UI', 12, 'bold'),
            fill='#fff'
        )
        
        # Initial angle
        self.detection_angle = 0
        
        # Start the animation
        self.animate_detection()
    
    def stop_detection_animation(self):
        """Stop the emotion detection animation"""
        if not hasattr(self, 'detection_animation_active') or not self.detection_animation_active:
            return
            
        self.detection_animation_active = False
        if hasattr(self, 'detection_animation_id') and self.detection_animation_id:
            self.root.after_cancel(self.detection_animation_id)
            self.detection_animation_id = None
            
        # Remove the animation canvas
        if hasattr(self, 'detection_canvas'):
            self.detection_canvas.destroy()
    
    def animate_detection(self):
        """Animate the detection dots in a rotating pattern"""
        if not hasattr(self, 'detection_animation_active') or not self.detection_animation_active:
            return
            
        # Update angle
        self.detection_angle += 10
        if self.detection_angle >= 360:
            self.detection_angle = 0
            
        # Move each dot based on updated angle
        center_x, center_y = 175, 90
        radius = 40
        
        for i, dot in enumerate(self.detection_dots):
            # Calculate new position with offset angle
            angle = self.detection_angle + i * (360 / len(self.detection_dots))
            x = center_x + radius * np.cos(np.radians(angle))
            y = center_y + radius * np.sin(np.radians(angle))
            
            # Get current size of dot
            x1, y1, x2, y2 = self.detection_canvas.coords(dot)
            size = (x2 - x1) / 2
            
            # Update position
            self.detection_canvas.coords(dot, x-size, y-size, x+size, y+size)
            
            # Pulse the dots - change color based on position
            brightness = 155 + int(50 * np.sin(np.radians(angle)))
            color = f'#{brightness:02x}{brightness//2:02x}{brightness:02x}'
            self.detection_canvas.itemconfig(dot, fill=color)
        
        # Update text with animated dots
        num_dots = (self.detection_angle // 45) % 4
        dots = "." * num_dots
        self.detection_canvas.itemconfig(self.detection_text, text=f"Processing{dots}")
        
        # Schedule next frame
        self.detection_animation_id = self.root.after(50, self.animate_detection)

def main():
    root = Tk()
    app = ModernEmotionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

if __name__ == "__main__":
    main()