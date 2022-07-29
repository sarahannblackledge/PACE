from moviepy.editor import VideoFileClip
import os

def video_to_gif(fpath_video):
    videoClip = VideoFileClip(fpath_video)
    fpath_gif = os.path.splitext(fpath_video[0] = '.gif')
    videoClip.write_gif(fpath_gif, fps=10)