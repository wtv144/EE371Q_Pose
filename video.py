
import subprocess
import io
import os
import shutil
import shlex
import json
from tkinter import *
from tkinter import filedialog

def find_video_properties(pathToInputVideo):
    cmd = "ffprobe -v quiet -print_format json -show_streams"
    args = shlex.split(cmd)
    args.append(pathToInputVideo)
    # run the ffprobe process, decode stdout into utf-8 & convert to JSON
    ffprobeOutput = subprocess.check_output(args).decode('utf-8')
    ffprobeOutput = json.loads(ffprobeOutput)

    # find height, width, and num frames (estimate)
    height = ffprobeOutput['streams'][0]['height']
    width = ffprobeOutput['streams'][0]['width']
    frames = ffprobeOutput['streams'][0]['nb_frames']
    duration = float(ffprobeOutput['streams'][0]['duration'])

    return height, width, frames, duration

def make_temp_folder():
    try:
        os.makedirs('temp_folder')
    except FileExistsError:
        shutil.rmtree(os.getcwd() + "/temp_folder")
        os.makedirs("temp_folder")
    os.chdir(os.getcwd() + "/temp_folder")
    return os.getcwd()


root = Tk()
root.filename = filedialog.askopenfilename(initialdir="/", title = "Select a File", filetypes=(("MP4 files", "*.mp4"),("MPEG4 files", "*.mpeg4"),("MOV files", "*.mov"),("AVI files", "*.avi"),("All Files", "*.*")))
filepath = root.filename
vid_h,vid_w, frames, duration = find_video_properties(filepath)

#subprocess.run(f'ffmpeg -y -i {filepath} -r 30 output.mp4')

dir_temp_folder = make_temp_folder()
subprocess.run(f'ffmpeg -r 1 -i {filepath} -r 1 ss%03d.jpg')

subprocess.run(f'ffmpeg -r 30 -i ss%03d.jpg -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -vcodec libx264 -y -an overlayoutput.mp4')
