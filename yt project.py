import os
import cv2
import numpy as np
import subprocess
import re
import tkinter as tk
from tkinter import messagebox, ttk
from subprocess import CREATE_NO_WINDOW
from concurrent.futures import ThreadPoolExecutor

class FrameExtractorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("⚡ Enhanced YouTube Frame Extractor")
        self.root.geometry("600x350")
        self.root.resizable(False, False)
        self.setup_ui()
        
        # Check dependencies
        self.check_dependencies()
        
    def setup_ui(self):
        # URL Entry
        tk.Label(self.root, text="📺 YouTube Video URL:", font=("Arial", 12)).pack(pady=5)
        self.url_entry = tk.Entry(self.root, width=65)
        self.url_entry.pack()
        
        # Frame Count
        tk.Label(self.root, text="🖼 Number of Frames to Extract:", font=("Arial", 12)).pack(pady=5)
        self.image_count_var = tk.StringVar(value="10")
        tk.Entry(self.root, textvariable=self.image_count_var, width=10, justify="center").pack()
        
        # Start Button
        tk.Button(self.root, text="Download & Extract HQ Frames", command=self.start_process,
                bg="#4CAF50", fg="white", font=("Arial", 12), padx=10, pady=5).pack(pady=10)
        
        # Progress Bar
        self.progress_bar = ttk.Progressbar(self.root, orient="horizontal", length=500, mode="determinate")
        self.progress_bar.pack(pady=5)
        
        # Status Label
        self.status_label = tk.Label(self.root, text="Ready", font=("Arial", 11), fg="gray")
        self.status_label.pack()
        
    def check_dependencies(self):
        try:
            subprocess.run(["yt-dlp", "--version"], check=True, capture_output=True, creationflags=CREATE_NO_WINDOW)
        except (subprocess.CalledProcessError, FileNotFoundError):
            messagebox.showerror("Error", "yt-dlp not found. Please install it from https://github.com/yt-dlp/yt-dlp")
            self.root.destroy()
            return

    @staticmethod
    def sanitize_filename(name):
        return re.sub(r'[\\/*?:"<>|]', "", name).strip()

    def get_video_title(self, link):
        try:
            result = subprocess.run(
                ["yt-dlp", "--get-title", "--no-playlist", link],
                capture_output=True,
                text=True,
                creationflags=CREATE_NO_WINDOW
            )
            if result.returncode != 0:
                raise Exception(f"Failed to get title: {result.stderr}")
            return self.sanitize_filename(result.stdout)
        except Exception as e:
            raise Exception(f"Error getting video title: {str(e)}")

    def download_video(self, link):
        output_folder = "downloaded_videos"
        os.makedirs(output_folder, exist_ok=True)
        
        title = self.get_video_title(link)
        filename = f"{title}.mp4"
        output_path = os.path.join(output_folder, filename)

        if os.path.exists(output_path):
            return output_path, title

        command = [
            "yt-dlp",
            "--no-playlist",
            "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
            "--merge-output-format", "mp4",
            "-o", output_path,
            "--quiet",
            link
        ]
        
        try:
            subprocess.run(command, check=True, creationflags=CREATE_NO_WINDOW)
            return output_path, title
        except subprocess.CalledProcessError as e:
            raise Exception(f"Video download failed: {str(e)}")

    def enhance_frame(self, frame):
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE for adaptive contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        merged = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        
        # Apply slight sharpening
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Final denoising
        return cv2.fastNlMeansDenoisingColored(sharpened, None, 10, 10, 7, 21)

    def process_frame(self, frame_idx, cap, output_folder, total_frames, total_images):
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                return False

            # Resize if needed (maintaining aspect ratio)
            if frame.shape[0] != 1080 or frame.shape[1] != 1920:
                frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)

            enhanced = self.enhance_frame(frame)
            frame_path = os.path.join(output_folder, f"frame_{frame_idx + 1}.jpg")
            cv2.imwrite(frame_path, enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            return True
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {str(e)}")
            return False

    def extract_frames(self, video_path, title, total_images=10):
        output_folder = os.path.join(os.path.dirname(video_path), "extracted_frames", title)
        os.makedirs(output_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Failed to open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_images > total_frames:
            total_images = total_frames

        # Select frames distributed throughout the video
        frame_indices = np.linspace(0, total_frames - 1, total_images, dtype=int)
        
        self.progress_bar["maximum"] = total_images
        self.progress_bar["value"] = 0
        self.status_label.config(text=f"⏳ Extracting {total_images} frames...")
        self.root.update()

        # Use threading for better performance
        successful_frames = 0
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for idx, frame_idx in enumerate(frame_indices):
                futures.append(executor.submit(
                    self.process_frame, frame_idx, cap, output_folder, total_frames, total_images
                ))
            
            for future in futures:
                if future.result():
                    successful_frames += 1
                self.progress_bar["value"] = successful_frames
                self.status_label.config(text=f"🖼 Processed {successful_frames}/{total_images} frames")
                self.root.update()

        cap.release()
        return output_folder, successful_frames

    def start_process(self):
        link = self.url_entry.get().strip()
        if not link:
            messagebox.showerror("Error", "Please enter a YouTube URL")
            return

        try:
            total_images = int(self.image_count_var.get())
            if total_images <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of frames (1 or more)")
            return

        try:
            self.status_label.config(text="⏳ Getting video information...")
            self.root.update()
            
            video_path, title = self.download_video(link)
            
            self.status_label.config(text="⏳ Extracting and enhancing frames...")
            self.root.update()
            
            output_folder, frame_count = self.extract_frames(video_path, title, total_images)
            
            self.status_label.config(text=f"✅ Success! {frame_count} frames extracted", fg="green")
            messagebox.showinfo("Complete", f"Successfully extracted {frame_count} frames to:\n{output_folder}")
            
            # Open the output folder
            if os.name == 'nt':
                os.startfile(output_folder)
            else:
                subprocess.run(['xdg-open', output_folder])
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
            self.status_label.config(text="❌ Process failed", fg="red")
            self.progress_bar["value"] = 0
        finally:
            self.root.update()

if __name__ == "__main__":
    root = tk.Tk()
    app = FrameExtractorApp(root)
    root.mainloop()
