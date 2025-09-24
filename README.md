

# ManoVision

**ManoVision** is a project that lets you interact with a virtual 3D environment using only your webcam. Your hands and gestures are tracked in real time and mapped into interactive 3D scenes.


## Demo




https://github.com/user-attachments/assets/9d8eb9df-8147-4500-b058-b89c2c19d2be




## Features

* Hand tracking with [MediaPipe](https://developers.google.com/mediapipe)
* Gesture-based interaction with virtual objects
* 3D rendering using [PyOpenGL](http://pyopengl.sourceforge.net/) and [GLFW](https://www.glfw.org/)
* Works with a standard laptop webcam
* Real-time visualization with [OpenCV](https://opencv.org/) and [Matplotlib](https://matplotlib.org/)
* Only **pinch gesture with right thumb and right index finger** rotates the cube
* Press **Esc** to quit


## Get Started

### Prerequisites

* Python 3.10
* A laptop with a webcam

### Installation

1. Clone the repository:

```bash
git clone https://github.com/Ashp116/ManoVision.git
cd ManoVision
```

2. Create a virtual environment (recommended):

```bash
python3.10 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the main script:

```bash
python main.py
```

* Your webcam will turn on.
* Use a **pinch gesture with your right thumb and right index finger** to rotate the cube in the virtual environment.
* Press **Esc** to quit.


## Requirements

See [requirements.txt](./requirements.txt) for the full list. Key dependencies:


* [mediapipe](https://pypi.org/project/mediapipe/)
* [opencv-python](https://pypi.org/project/opencv-python/)
 
* [PyOpenGL](https://pypi.org/project/PyOpenGL/)
* [glfw](https://pypi.org/project/glfw/)
* [pygame](https://pypi.org/project/pygame/)
* [matplotlib](https://pypi.org/project/matplotlib/)
* [numpy](https://pypi.org/project/numpy/)
* [scipy](https://pypi.org/project/scipy/)

Other dependencies are included in the `requirements.txt` file.
