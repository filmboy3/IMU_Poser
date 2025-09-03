# QuantumLeap Validator iOS App

## 🎯 Mission
Bridge the sim-to-real gap for the QuantumLeap Pose Engine by capturing synchronized IMU + Vision data from real-world exercise sessions.

## 🏆 Achievement Context
Built on the foundation of our **breakthrough MPJPE of 0.0228** - this app is the critical tool for validating and improving our production-grade pose estimation technology in real-world conditions.

## 📱 App Features

### 1. Live Feedback Mode (Vision-First)
- **Real-time pose visualization** using Apple's Vision framework
- **Instant skeletal overlay** on camera feed during exercise
- **High-quality user experience** with immediate visual feedback

### 2. Data Capture Mode (IMU + Vision)
- **Synchronized data recording**: 100Hz IMU + 30fps video
- **Perfect timestamp alignment** for multi-modal training
- **Background capture** while user enjoys live feedback

### 3. Session Management & Export
- **Structured data storage** in device documents directory
- **One-tap export** via iOS Share Sheet (.zip format)
- **Session metadata** with exercise type, duration, and quality metrics

## 🛠️ Technical Architecture

### Core Components

#### MotionManager.swift
- **100Hz IMU data stream** using CoreMotion
- **Combine publishers** for reactive data flow
- **Background queue processing** for optimal performance
- **Real-time data rate monitoring**

#### PoseViewController.swift
- **AVFoundation camera capture** with front-facing camera
- **Vision framework integration** for real-time pose detection
- **Custom overlay rendering** with skeletal visualization
- **Sample buffer processing** for video recording

#### PoseOverlayView.swift
- **Real-time skeletal drawing** with color-coded body parts
- **Confidence-based filtering** (threshold: 0.3)
- **Coordinate system conversion** (Vision → UIView)
- **Pose quality assessment** and validation

#### SessionRecorder.swift
- **Multi-stream data synchronization** (IMU + Video + Pose)
- **HDF5-compatible data structuring** for model integration
- **Automatic session management** with unique IDs
- **Export functionality** with zip compression

### Data Flow Architecture
```
iPhone Sensors → MotionManager → SessionRecorder
     ↓                              ↓
Camera Feed → PoseViewController → Video Recording
     ↓                              ↓
Vision API → PoseOverlayView → Pose Data
                                   ↓
                            Synchronized Export
```

## 📊 Data Output Format

### Session Structure
```
session_YYYYMMDD_HHMMSS_UUID/
├── video.mp4                 # 1080p H.264 video
├── imu_data.json            # 100Hz IMU samples
├── pose_data.json           # Vision framework poses
└── session_metadata.json    # Session info & metrics
```

### IMU Data Format
```json
{
  "timestamp": 1234567890.123,
  "acceleration": {"x": 0.1, "y": -9.8, "z": 0.2},
  "rotationRate": {"x": 0.01, "y": 0.02, "z": 0.03},
  "attitude": {
    "roll": 0.1, "pitch": 0.2, "yaw": 0.3,
    "quaternionX": 0.1, "quaternionY": 0.2, 
    "quaternionZ": 0.3, "quaternionW": 0.9
  }
}
```

### Pose Data Format
```json
{
  "nose": {"x": 0.5, "y": 0.3, "confidence": 0.95},
  "left_shoulder": {"x": 0.4, "y": 0.4, "confidence": 0.87},
  "right_shoulder": {"x": 0.6, "y": 0.4, "confidence": 0.89}
  // ... all 17 body landmarks
}
```

## 🔄 Data Processing Pipeline

### Python Integration (`data_ingestion_pipeline.py`)
- **Automated zip file processing** for exported sessions
- **MediaPipe pose extraction** from video files
- **Multi-modal data synchronization** using timestamps
- **HDF5 dataset creation** compatible with QuantumLeap model
- **Batch processing** for multiple sessions

### Usage
```bash
# Process single session
python data_ingestion_pipeline.py session_20250903_143022_A1B2C3D4.zip

# Process all sessions in directory
python data_ingestion_pipeline.py /path/to/sessions/ --create-dataset

# Output: quantumleap_real_world_dataset.h5
```

## 🚀 Getting Started

### Prerequisites
- **Xcode 15.0+** with iOS 17.0+ deployment target
- **iPhone with front-facing camera** and motion sensors
- **Python 3.8+** with dependencies for data processing

### Installation
1. **Open project** in Xcode: `QuantumLeapValidator.xcodeproj`
2. **Configure signing** with your Apple Developer account
3. **Build and run** on physical device (camera required)
4. **Install Python dependencies**:
   ```bash
   pip install opencv-python mediapipe h5py pandas numpy
   ```

### First Session
1. **Launch app** and select exercise type (Squat, Bicep Curl, etc.)
2. **Position phone** on tripod or lean against wall
3. **Start session** - live pose overlay appears
4. **Begin recording** when ready to capture data
5. **Stop recording** and export session via Share Sheet
6. **Process data** using Python pipeline

## 📈 Real-World Validation Strategy

### Phase 1: Data Collection (Weeks 1-2)
- [ ] Capture 100+ squat sessions across different users
- [ ] Validate data quality and synchronization
- [ ] Compare Vision poses with QuantumLeap predictions

### Phase 2: Model Adaptation (Weeks 3-4)
- [ ] Fine-tune QuantumLeap model on real IMU data
- [ ] Implement held-phone domain adaptation
- [ ] Validate sim-to-real transfer performance

### Phase 3: Multi-Modal Fusion (Weeks 5-8)
- [ ] Train Vision+IMU fusion model
- [ ] Achieve >95% pose accuracy on real data
- [ ] Deploy production inference pipeline

## 🎯 Success Metrics

### Technical Targets
- **Data Capture Rate**: >95% successful sessions
- **Synchronization Accuracy**: <10ms timestamp drift
- **Pose Detection Rate**: >90% frames with valid pose
- **Export Success**: 100% reliable data export

### Model Performance Goals
- **Real-World MPJPE**: <0.05 (vs 0.0228 synthetic)
- **Domain Transfer**: <20% performance degradation
- **Multi-Modal Fusion**: >10% improvement over single modality

## 🔬 Research Applications

### Sim-to-Real Analysis
- **Domain gap quantification** between MuJoCo and real data
- **Transfer learning effectiveness** measurement
- **Robustness evaluation** across user demographics

### Multi-Modal Learning
- **Vision-IMU complementarity** analysis
- **Sensor fusion architecture** optimization
- **Real-time inference** performance benchmarking

## 🛡️ Privacy & Security

### Data Protection
- **Local storage only** - no cloud transmission
- **User-controlled export** via standard iOS sharing
- **Automatic cleanup** of temporary processing files
- **No biometric data retention** beyond session scope

### Permissions
- **Camera access**: Real-time pose visualization
- **Motion sensors**: IMU data capture for research
- **File system**: Local session storage and export

## 🎉 Impact & Vision

This iOS Validator App represents the **critical bridge** between our breakthrough synthetic training success and real-world deployment. By capturing high-quality, synchronized multi-modal data, we're building the foundation for:

- **Production-grade pose estimation** in mobile applications
- **Personalized fitness coaching** with real-time feedback
- **Scalable data collection** for continuous model improvement
- **Research advancement** in human movement analysis

**From 0.0228 MPJPE to Real-World Impact** - this is how we make it happen.

---

*Built with the precision of our physics-first approach and the ambition to revolutionize human movement analysis.*
