# QuantumLeap Validator - Deployment Guide

## 🚀 Quick Start Deployment

### Prerequisites Checklist
- [ ] **Xcode 15.0+** installed on macOS
- [ ] **Apple Developer Account** (free or paid)
- [ ] **iPhone with iOS 17.0+** for testing
- [ ] **Python 3.8+** for data processing
- [ ] **Physical device required** (camera + motion sensors)

### 1. iOS App Deployment

#### Step 1: Open Project
```bash
cd /Users/jonathanschwartz/Documents/Social_Content/QuantumLeapValidator
open QuantumLeapValidator.xcodeproj
```

#### Step 2: Configure Signing
1. Select **QuantumLeapValidator** target in Xcode
2. Go to **Signing & Capabilities** tab
3. Set **Team** to your Apple Developer account
4. Ensure **Bundle Identifier** is unique (e.g., `com.yourname.quantumleap.validator`)

#### Step 3: Build and Deploy
1. Connect iPhone via USB
2. Select your device in Xcode's device selector
3. Press **⌘+R** to build and run
4. **Trust developer** on device when prompted

### 2. Python Environment Setup

#### Install Dependencies
```bash
# Create virtual environment
python -m venv quantumleap_env
source quantumleap_env/bin/activate  # On macOS/Linux

# Install core dependencies
pip install numpy pandas h5py

# Optional: Install computer vision dependencies
pip install opencv-python mediapipe
```

#### Test Data Pipeline
```bash
cd QuantumLeapValidator
python test_data_pipeline_simple.py
```

Expected output:
```
✅ IMU samples: 1000
✅ IMU acceleration shape: (1000, 3)  
✅ Pose frames: 300
🎉 All tests passed successfully!
```

## 📱 Using the iOS App

### First Session Workflow

#### 1. Launch App
- Open **QuantumLeap Validator** on iPhone
- Grant **Camera** and **Motion** permissions when prompted

#### 2. Setup Exercise Session
- Select exercise type: **Squat**, **Bicep Curl**, **Lateral Raise**, or **Push-up**
- Tap **"Start [Exercise] Session"**

#### 3. Position Device
- **Place iPhone on tripod** or lean against wall
- **Front camera should face you** with full body visible
- **Ensure good lighting** for pose detection

#### 4. Record Session
- **Green skeletal overlay** appears when pose detected
- Tap **"Record"** button to start data capture
- **Red "REC" indicator** shows active recording
- **Perform 5-10 repetitions** of selected exercise
- Tap **"Stop"** when finished

#### 5. Export Data
- Tap **"Export"** button after recording
- Use **iOS Share Sheet** to save/send data
- **ZIP file contains**: video.mp4, imu_data.json, pose_data.json, metadata

### Session Quality Indicators

#### Good Session Markers
- ✅ **Green pose overlay** visible throughout
- ✅ **IMU rate >90 Hz** shown in top-right
- ✅ **Smooth skeletal tracking** without jumps
- ✅ **Full body visible** in camera frame

#### Poor Session Markers  
- ❌ **Red IMU indicator** or low Hz rate
- ❌ **Flickering pose overlay** or no detection
- ❌ **Partial body visibility** or poor lighting
- ❌ **Excessive device movement** during recording

## 🔄 Data Processing Pipeline

### Process Exported Sessions

#### Single Session Processing
```bash
# Process one exported session
python data_ingestion_pipeline.py session_20250903_143022_A1B2C3D4.zip

# Output: session_20250903_143022_A1B2C3D4_processed.h5
```

#### Batch Processing
```bash
# Process all sessions in directory
python data_ingestion_pipeline.py /path/to/exported/sessions/ --create-dataset

# Output: quantumleap_real_world_dataset.h5
```

### Data Validation
```bash
# Validate processed data
python -c "
import h5py
with h5py.File('session_processed.h5', 'r') as f:
    print('Groups:', list(f.keys()))
    print('IMU samples:', len(f['imu_data/timestamps']))
    print('Pose frames:', len(f['pose_data/timestamps']))
"
```

## 🔧 Troubleshooting

### Common iOS Issues

#### App Won't Build
**Error**: Code signing issues
**Solution**: 
1. Check Apple Developer account in Xcode preferences
2. Ensure unique Bundle Identifier
3. Clean build folder (⌘+Shift+K)

#### Camera Not Working
**Error**: Black screen or no camera feed
**Solution**:
1. Check camera permissions in iOS Settings
2. Restart app
3. Try different lighting conditions

#### Pose Detection Failing
**Error**: No skeletal overlay visible
**Solution**:
1. Ensure full body visible in frame
2. Improve lighting conditions
3. Remove background clutter
4. Stand 3-6 feet from camera

#### IMU Data Rate Low
**Error**: IMU showing <90 Hz
**Solution**:
1. Close other apps to free resources
2. Restart iPhone
3. Ensure iOS 17.0+ installed

### Common Python Issues

#### Import Errors
**Error**: `ModuleNotFoundError: No module named 'cv2'`
**Solution**:
```bash
pip install opencv-python mediapipe
# OR use simplified pipeline:
python test_data_pipeline_simple.py
```

#### HDF5 File Corruption
**Error**: Cannot read processed .h5 files
**Solution**:
```bash
# Check file integrity
python -c "import h5py; h5py.File('file.h5', 'r').keys()"

# Reprocess from original zip if corrupted
python data_ingestion_pipeline.py original_session.zip
```

## 📊 Data Quality Assessment

### Session Quality Metrics

#### Excellent Session (Target)
- **Duration**: 30-60 seconds
- **IMU Rate**: >95 Hz consistent
- **Pose Detection**: >90% frames with valid pose
- **Movement Quality**: Smooth, controlled repetitions
- **Data Sync**: <10ms timestamp drift

#### Acceptable Session (Usable)
- **Duration**: 15-90 seconds  
- **IMU Rate**: >85 Hz average
- **Pose Detection**: >70% frames with valid pose
- **Movement Quality**: Some variation acceptable
- **Data Sync**: <50ms timestamp drift

#### Poor Session (Discard)
- **Duration**: <10 seconds or >120 seconds
- **IMU Rate**: <80 Hz or frequent dropouts
- **Pose Detection**: <50% frames with valid pose
- **Movement Quality**: Erratic or incomplete movements
- **Data Sync**: >100ms timestamp drift

### Automated Quality Checks
```python
# Quality assessment script
def assess_session_quality(h5_file):
    with h5py.File(h5_file, 'r') as f:
        imu_rate = len(f['imu_data/timestamps']) / f.attrs['duration']
        pose_rate = len(f['pose_data/timestamps']) / f.attrs['duration']
        
        quality_score = 0
        if imu_rate > 95: quality_score += 40
        elif imu_rate > 85: quality_score += 30
        elif imu_rate > 80: quality_score += 20
        
        if pose_rate > 25: quality_score += 40
        elif pose_rate > 20: quality_score += 30
        elif pose_rate > 15: quality_score += 20
        
        duration = f.attrs['duration']
        if 30 <= duration <= 60: quality_score += 20
        elif 15 <= duration <= 90: quality_score += 10
        
        return quality_score  # 0-100 scale
```

## 🎯 Production Deployment Checklist

### Pre-Deployment Validation
- [ ] **App builds successfully** on multiple devices
- [ ] **Camera permissions** granted and functional
- [ ] **Motion sensors** working at >90 Hz
- [ ] **Pose detection** working in various lighting
- [ ] **Data export** creates valid ZIP files
- [ ] **Python pipeline** processes exported data
- [ ] **HDF5 output** compatible with QuantumLeap model

### Data Collection Protocol
- [ ] **Standardized setup**: Consistent camera positioning
- [ ] **Exercise guidelines**: Clear movement instructions
- [ ] **Session duration**: 30-60 seconds optimal
- [ ] **Quality thresholds**: Automated filtering criteria
- [ ] **Batch processing**: Efficient multi-session handling
- [ ] **Data validation**: Integrity checks before training

### Integration with QuantumLeap Model
- [ ] **Data format compatibility** with existing training pipeline
- [ ] **Coordinate system alignment** between Vision and MuJoCo
- [ ] **Temporal synchronization** for multi-modal fusion
- [ ] **Domain adaptation** strategy for sim-to-real transfer
- [ ] **Performance benchmarking** against synthetic data

This deployment guide ensures smooth transition from development to production data collection, enabling the critical sim-to-real validation phase of the QuantumLeap Pose Engine.
