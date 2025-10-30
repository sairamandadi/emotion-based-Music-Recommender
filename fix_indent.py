import re

# Read the original file
with open('Emotion.py', 'r', encoding='utf-8') as file:
    content = file.read()

# Find the next_song method and fix the indentation
pattern = r'def next_song\(self\):(.*?)def shuffle_song'
replacement = r'''def next_song(self):
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

    def shuffle_song'''

fixed_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Write the fixed content to a new file
with open('Emotion_fixed.py', 'w', encoding='utf-8') as file:
    file.write(fixed_content)

print("Fixed file created as Emotion_fixed.py") 