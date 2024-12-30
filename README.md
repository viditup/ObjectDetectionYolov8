# Object Detection 4x - Project Overview

Welcome to the **Object Detection 4x** repository. This project demonstrates various object detection applications using YOLO (You Only Look Once) and other models. It includes real-world projects such as car counting, people detection, PPE (Personal Protective Equipment) detection, and poker hand recognition. This repository provides tools for both video-based detection and real-time webcam object detection, leveraging CUDA and PyTorch for GPU acceleration.

### **Key Projects in the Repository:**
- **Chapter 3 - Video Detection**: Object detection applied to video files.
- **Chapter 5 - Running YOLO**: Setting up and running YOLO for object detection.
- **Chapter 6 - YOLO with Webcam**: Implementing YOLO for real-time object detection using webcam input.
- **Project 1 - Car Counter**: Counting cars in real-time video streams.
- **Project 1 - Car Counter (City)**: Car counting applied in an urban setting with city-specific adjustments.
- **Project 2 - People Counter (Elevator)**: Detecting and counting people in an elevator.
- **Project 3 - PPE Detection**: Detecting whether individuals are wearing personal protective equipment.
- **Project 4 - Poker Hand Detector**: Detecting poker hands from images/videos.
- **YOLO Weights**: Pre-trained weights for YOLO models.

---

### **Project Showcases**

#### **Chapter 3 - Video Detection**

In this chapter, the focus is on detecting objects in video files. The YOLO model is used to process each frame and detect multiple objects in real-time.

![Screenshot 2024-12-30 181913](https://github.com/user-attachments/assets/f49749d7-9dcd-426a-a01e-9a8e2b8bf2e8)

#### **Chapter 5 - Running YOLO**

Here we demonstrate how to set up YOLO for object detection. The provided script loads pre-trained YOLO models and runs object detection on images and videos.

![Screenshot 2024-12-30 181940](https://github.com/user-attachments/assets/9873a84b-d5d9-473d-84f1-e98962422ff8)

#### **Chapter 6 - YOLO with Webcam**

This chapter showcases how to perform real-time object detection using a webcam. YOLO processes the live feed, identifying objects and displaying the results in real time.

![Screenshot 2024-12-30 181426](https://github.com/user-attachments/assets/eff8558a-2da3-4214-a167-a8a3eef52a42)

#### **Project 1 - Car Counter**

This project detects cars in real-time and counts them as they pass through a specific area. It is suitable for traffic monitoring and parking lot surveillance.

![Screenshot 2024-12-30 180714](https://github.com/user-attachments/assets/4c5e8417-2633-4472-ab75-83f0e37be028)
![Screenshot 2024-12-30 180735](https://github.com/user-attachments/assets/ebd680d3-77ab-4113-a8ea-e54306d05081)
![Screenshot 2024-12-30 180803](https://github.com/user-attachments/assets/edcbc82c-bcc8-46df-9634-eb9f6988e844)

#### **Project 1 - Car Counter (City)**

A city-specific variation of the car counter project, optimized for detecting cars in urban environments with more complex backgrounds and varying lighting conditions.

![Screenshot 2024-12-30 182354](https://github.com/user-attachments/assets/7d9584a8-811c-4068-b1d1-4f06cecfc1f8)
![Screenshot 2024-12-30 182400](https://github.com/user-attachments/assets/d3d72d66-75da-4f8f-a7d2-a96a95787a49)

#### **Project 2 - People Counter (Elevator)**

This project uses object detection to count the number of people entering or exiting an elevator. It can be adapted for other environments that require real-time person counting.

![Screenshot 2024-12-30 180834](https://github.com/user-attachments/assets/a44bdabc-ebb5-4a9f-94bc-3fe423ba55fb)
![Screenshot 2024-12-30 180947](https://github.com/user-attachments/assets/b027813f-2cfd-4d21-9c65-7365a6204110)
![Screenshot 2024-12-30 181036](https://github.com/user-attachments/assets/b5b9cf26-aab1-4f11-9570-10987ca96d23)
![Screenshot 2024-12-30 181202](https://github.com/user-attachments/assets/8dd97239-2a90-46be-bae5-abac399bbbca)

#### **Project 3 - PPE Detection**

This project is designed to identify whether people are wearing appropriate personal protective equipment, such as helmets, vests, or face masks. It's useful in safety-critical environments like construction sites or factories.

![Screenshot 2024-12-30 181512](https://github.com/user-attachments/assets/6398962d-4c42-4d1a-ab64-f1cf44f38463)
![Screenshot 2024-12-30 181410](https://github.com/user-attachments/assets/c0981a11-065b-434b-b52e-6d2757f8d27e)

#### **Project 4 - Poker Hand Detector**

This application detects poker hands in images or video streams, aiding in automated poker games or card recognition systems.

![Screenshot 2024-12-30 181544](https://github.com/user-attachments/assets/8a429111-0118-4dbd-a65b-7b7bb42f3e07)
![Screenshot 2024-12-30 181637](https://github.com/user-attachments/assets/a860b854-1744-40b4-a4ce-001b71a399ee)



---

### **Tech Stack and Dependencies**

This project uses several libraries for object detection, including PyTorch for deep learning and CUDA for GPU acceleration to speed up the inference process.

#### **Required Libraries:**

- **cvzone**: 1.5.6
- **ultralytics**: 8.0.26 (for YOLO models)
- **hydra-core**: >=1.2.0
- **matplotlib**: >=3.2.2
- **numpy**: >=1.18.5
- **opencv-python**: 4.5.4.60
- **Pillow**: >=7.1.2
- **PyYAML**: >=5.3.1
- **requests**: >=2.23.0
- **scipy**: >=1.4.1
- **torch**: >=1.7.0 (for deep learning)
- **torchvision**: >=0.8.1
- **tqdm**: >=4.64.0
- **filterpy**: 1.4.5 (for Kalman filtering)
- **scikit-image**: 0.19.3
- **lap**: 0.4.0 (for linear assignment problems)

#### **Environment Setup:**

1. **CUDA & GPU Support**: This project uses CUDA for GPU acceleration. Ensure you have a CUDA-compatible GPU and have installed the appropriate drivers.
   
2. **PyTorch with CUDA**: To leverage GPU acceleration, install PyTorch with CUDA support:
   ```bash
   pip install torch torchvision torchaudio
   ```
   Make sure to install the correct version based on your system's CUDA version. For example, for CUDA 11.0:
   ```bash
   pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/cuda/11.0/torch_stable.html
   ```

---

### **Project Structure:**
```bash
Object-Detection-4x/
├── Chapter 3 - Video Detection/
├── Chapter 5 - Running YOLO/
├── Chapter 6 - Yolo with Webcam/
├── Project 1 - Car Counter/
├── Project 1 - Car Counter City/
├── Project 2 - People Counter Elevator/
├── Project 3 - PPE Detection/
├── Project 4 - Poker Hand Detector/
├── Videos/
├── Yolo-Weights/
└── README.md
```

### **How to Run the Code**

1. **Install Dependencies**: Install the required dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```

2. **Choose a Project**: Navigate to the folder of the project you wish to run. For example:
   ```bash
   cd Project 1 - Car Counter
   ```

3. **Run the Script**: Each project has a main Python script (e.g., `car_counter.py`, `ppe_detection.py`) that you can execute:
   ```bash
   python <script_name>.py
   ```

4. **Webcam or Video Input**: For real-time detection via webcam, ensure your webcam is connected and run the appropriate script under "YOLO with Webcam" or similar.

5. **YOLO Model Weights**: If you are using YOLO, ensure the correct weights file (`yolov5s.pt`, etc.) is available in the `/Yolo-Weights/` directory.

---

### **Contributing**

Feel free to contribute! To submit your contributions:

1. Fork the repository.
2. Create a feature branch.
3. Implement your changes.
4. Submit a pull request with a clear description of your changes.

---

### **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### **Acknowledgments**

- Special thanks to the authors and contributors of the YOLO model.
- Thanks to the maintainers of the libraries used in this project, including OpenCV, PyTorch, and others.
