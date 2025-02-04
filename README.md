## Object Detection v1.0.0

## Overview
This project implements an object detection system using YOLOv5 and OpenCV. It detects objects in real-time from a webcam feed, displaying relevant information such as FPS and detected objects. The system also allows users to record video and capture frames on demand using a pre-trained YOLOv5 model from the Ultralytics repository.

## Features
- **Real-time Object Detection**: Detects and classifies objects in the camera stream.
- **FPS Display**: Shows real-time FPS to monitor performance.
- **Recording**: Toggle video recording with the press of a button. Saves the video in **MP4** format.
- **Frame Capture**: Capture and save frames as images with a timestamped filename.
- **Alert on Detection**: Highlights the number of detected objects and alerts the user.

## Controls
- **Press 'q'**: Quit the program.
- **Press 'r'**: Toggle video recording on/off.
- **Press 'c'**: Capture the current frame as a JPG image.

## Example
![TestCase](https://github.com/pathanin-kht/Object-Detector/blob/5a727539c3cfdb7185cd8b608a984dc9b60506a4/TestCase.jpg)
  
### Installation
1. Clone or download the repository.
   ```bash
   git clone https://github.com/pathanin-kht/ObjectDetector.git
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
4. Run the script.
   ```bash
   python object_detection.py

## Acknowledgements
- **YOLOv5** for object detection [Ultralytics](https://github.com/ultralytics/yolov5)
- **OpenCV** for video handling [OpenCV](https://opencv.org/)
- **PyTorch** for deep learning [Pytorch](https://pytorch.org/)
- **NumPy** for numerical computation [Numpy](https://numpy.org/)

## License
This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

## Contact
For feedback or inquiries, feel free to reach out via [pathanin.kht@gmail.com](pathanin.kht@gmail.com).
