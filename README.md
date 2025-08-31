# YOLO11-Rotonda

YOLO11-Rotonda is a Python-based project for real-time roundabout (rotonda) detection in traffic scenes using the YOLO (You Only Look Once) deep learning architecture. This repository provides a modular pipeline for training, evaluating, and deploying object detection models specifically tailored for roundabout identification and analysis, supporting intelligent transportation systems and autonomous driving applications.

## Features

- **State-of-the-Art Object Detection:** Utilizes the YOLO architecture for high-speed, accurate detection of roundabouts in diverse traffic environments.
- **End-to-End Pipeline:** Includes data preprocessing, model training, inference, and evaluation scripts.
- **Custom Dataset Support:** Easily integrate your own labeled roundabout datasets.
- **Real-Time Inference:** Optimized for deployment in edge and real-time systems.
- **Modular & Extensible:** Clean codebase following Python best practices for easy adaptation and expansion.

## Use Cases

- **Autonomous Driving:** Enhance vehicle situational awareness by reliably detecting roundabouts.
- **Traffic Analysis:** Support urban planning and infrastructure monitoring with automated detection.
- **Research & Experimentation:** Serve as a baseline for developing advanced object detection models in traffic domains.

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch (recommended)
- OpenCV
- NumPy
- Other dependencies as listed in `requirements.txt`

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ignacio-serrano-rodriguez/yolo11-rotonda.git
    cd yolo11-rotonda
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

#### Training

Train your YOLO model on a roundabout dataset:

```bash
python train.py --data data/rotonda.yaml --cfg configs/yolovX.yaml --weights '' --batch-size 16
```

#### Inference

Run inference on images or video streams:

```bash
python detect.py --weights runs/train/exp/weights/best.pt --source path/to/images/
```

#### Evaluation

Evaluate model performance on a test set:

```bash
python test.py --data data/rotonda.yaml --weights runs/train/exp/weights/best.pt
```

## Project Structure

```
.
├── data/           # Dataset configuration and samples
├── models/         # YOLO model definitions and configs
├── scripts/        # Utility and helper scripts
├── runs/           # Training outputs and logs
├── train.py        # Training entrypoint
├── detect.py       # Inference entrypoint
├── test.py         # Evaluation script
├── requirements.txt
└── README.md
```

## Customization

- **Dataset:** Update `data/rotonda.yaml` to point to your custom dataset.
- **Model Config:** Modify or add configs in `models/` as needed.
- **Hyperparameters:** Adjust training parameters in `train.py` or config files.

## Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes, improvements, or new features.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a pull request

## Acknowledgements

- YOLO authors and the open-source community
- PyTorch, OpenCV, and related libraries

## Contact

For questions, collaboration, or support, please contact [Ignacio Serrano Rodriguez](https://github.com/ignacio-serrano-rodriguez).
