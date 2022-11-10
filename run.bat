title YOLOv7 Edged
if not DEFINED IS_MINIMIZED set IS_MINIMIZED=1 && start "" /min "%~dpnx0" %* && exit
@echo off
echo "YOLOv7 edged starting..."
call "C:\ProgramData\Anaconda3\Scripts\activate.bat" yolov7-edged
python api.py
exit