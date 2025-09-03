import UIKit
import AVFoundation
import Vision
import SwiftUI

class PoseViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    private var captureSession: AVCaptureSession!
    private var videoPreviewLayer: AVCaptureVideoPreviewLayer!
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private let visionSequenceHandler = VNSequenceRequestHandler()
    
    // Add a custom view for drawing the pose overlay
    private var poseOverlayView: PoseOverlayView!
    
    // Recording state
    var isRecording = false
    var onRecordingStateChanged: ((Bool) -> Void)?
    var selectedExercise: String = "Squat"
    
    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        
        setupCamera()
        
        // Setup the overlay view
        poseOverlayView = PoseOverlayView(frame: view.bounds)
        poseOverlayView.backgroundColor = .clear
        view.addSubview(poseOverlayView)
        
        // Add close button
        setupCloseButton()
        
        // Add recording indicator
        setupRecordingIndicator()
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
        DispatchQueue.global(qos: .userInitiated).async {
            if !self.captureSession.isRunning {
                self.captureSession.startRunning()
            }
        }
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        
        DispatchQueue.global(qos: .userInitiated).async {
            if self.captureSession.isRunning {
                self.captureSession.stopRunning()
            }
        }
    }
    
    private func setupCamera() {
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .high
        
        guard let captureDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front),
              let input = try? AVCaptureDeviceInput(device: captureDevice) else {
            print("Error: Could not create video device input.")
            return
        }
        
        if captureSession.canAddInput(input) {
            captureSession.addInput(input)
        }
        
        videoPreviewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        videoPreviewLayer.frame = view.layer.bounds
        videoPreviewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(videoPreviewLayer)
        
        // Configure video output
        videoDataOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
        ]
        videoDataOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue", qos: .userInitiated))
        videoDataOutput.alwaysDiscardsLateVideoFrames = true
        
        if captureSession.canAddOutput(videoDataOutput) {
            captureSession.addOutput(videoDataOutput)
        }
        
        // Set video orientation
        if let connection = videoDataOutput.connection(with: .video) {
            connection.videoOrientation = .portrait
            if connection.isVideoMirroringSupported {
                connection.isVideoMirrored = true
            }
        }
    }
    
    private func setupCloseButton() {
        let closeButton = UIButton(type: .system)
        closeButton.setImage(UIImage(systemName: "xmark.circle.fill"), for: .normal)
        closeButton.tintColor = .white
        closeButton.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        closeButton.layer.cornerRadius = 20
        closeButton.translatesAutoresizingMaskIntoConstraints = false
        closeButton.addTarget(self, action: #selector(closeButtonTapped), for: .touchUpInside)
        
        view.addSubview(closeButton)
        
        NSLayoutConstraint.activate([
            closeButton.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            closeButton.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            closeButton.widthAnchor.constraint(equalToConstant: 40),
            closeButton.heightAnchor.constraint(equalToConstant: 40)
        ])
    }
    
    private func setupRecordingIndicator() {
        let recordingLabel = UILabel()
        recordingLabel.text = "● REC"
        recordingLabel.textColor = .red
        recordingLabel.font = UIFont.systemFont(ofSize: 16, weight: .bold)
        recordingLabel.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        recordingLabel.textAlignment = .center
        recordingLabel.layer.cornerRadius = 8
        recordingLabel.clipsToBounds = true
        recordingLabel.translatesAutoresizingMaskIntoConstraints = false
        recordingLabel.alpha = 0 // Initially hidden
        
        view.addSubview(recordingLabel)
        
        NSLayoutConstraint.activate([
            recordingLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            recordingLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            recordingLabel.widthAnchor.constraint(equalToConstant: 60),
            recordingLabel.heightAnchor.constraint(equalToConstant: 30)
        ])
        
        // Store reference for animation
        recordingLabel.tag = 999
    }
    
    @objc private func closeButtonTapped() {
        dismiss(animated: true)
    }
    
    func startRecording() {
        isRecording = true
        onRecordingStateChanged?(true)
        
        // Show recording indicator with animation
        if let recordingLabel = view.viewWithTag(999) {
            UIView.animate(withDuration: 0.3) {
                recordingLabel.alpha = 1.0
            }
            
            // Add blinking animation
            let blinkAnimation = CABasicAnimation(keyPath: "opacity")
            blinkAnimation.duration = 1.0
            blinkAnimation.repeatCount = .infinity
            blinkAnimation.autoreverses = true
            blinkAnimation.fromValue = 1.0
            blinkAnimation.toValue = 0.3
            recordingLabel.layer.add(blinkAnimation, forKey: "blink")
        }
        
        print("PoseViewController: Started recording \(selectedExercise)")
    }
    
    func stopRecording() {
        isRecording = false
        onRecordingStateChanged?(false)
        
        // Hide recording indicator
        if let recordingLabel = view.viewWithTag(999) {
            recordingLabel.layer.removeAnimation(forKey: "blink")
            UIView.animate(withDuration: 0.3) {
                recordingLabel.alpha = 0.0
            }
        }
        
        print("PoseViewController: Stopped recording")
    }
    
    // MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        
        // Create Vision request for human body pose detection
        let humanBodyPoseRequest = VNDetectHumanBodyPoseRequest { [weak self] (request, error) in
            guard let self = self else { return }
            
            if let error = error {
                print("Vision request error: \(error.localizedDescription)")
                return
            }
            
            guard let results = request.results as? [VNHumanBodyPoseObservation] else {
                return
            }
            
            DispatchQueue.main.async {
                // Pass the results to our overlay view to be drawn
                self.poseOverlayView.processObservations(results)
            }
        }
        
        // Set confidence threshold
        humanBodyPoseRequest.revision = VNDetectHumanBodyPoseRequestRevision1
        
        do {
            try visionSequenceHandler.perform([humanBodyPoseRequest], on: sampleBuffer)
        } catch {
            print("Failed to perform Vision request: \(error)")
        }
        
        // If recording, pass sample buffer to session recorder
        if isRecording {
            SessionRecorder.shared.processSampleBuffer(sampleBuffer)
        }
    }
}

// MARK: - SwiftUI Integration

struct PoseViewControllerWrapper: UIViewControllerRepresentable {
    let selectedExercise: String
    @State private var isRecording = false
    @Environment(\.dismiss) private var dismiss
    
    func makeUIViewController(context: Context) -> PoseViewController {
        let controller = PoseViewController()
        controller.selectedExercise = selectedExercise
        controller.onRecordingStateChanged = { recording in
            isRecording = recording
        }
        return controller
    }
    
    func updateUIViewController(_ uiViewController: PoseViewController, context: Context) {
        // Update if needed
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject {
        let parent: PoseViewControllerWrapper
        
        init(_ parent: PoseViewControllerWrapper) {
            self.parent = parent
        }
    }
}

// MARK: - SwiftUI Overlay Controls

struct PoseControlsOverlay: View {
    @Binding var isRecording: Bool
    let selectedExercise: String
    let onStartRecording: () -> Void
    let onStopRecording: () -> Void
    let onDismiss: () -> Void
    
    var body: some View {
        VStack {
            // Top controls
            HStack {
                Button(action: onDismiss) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title)
                        .foregroundColor(.white)
                        .background(Color.black.opacity(0.6))
                        .clipShape(Circle())
                }
                
                Spacer()
                
                if isRecording {
                    HStack {
                        Circle()
                            .fill(Color.red)
                            .frame(width: 8, height: 8)
                        Text("REC")
                            .font(.caption)
                            .fontWeight(.bold)
                    }
                    .foregroundColor(.red)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(Color.black.opacity(0.6))
                    .cornerRadius(15)
                }
            }
            .padding()
            
            Spacer()
            
            // Bottom controls
            VStack(spacing: 20) {
                Text(selectedExercise)
                    .font(.title2)
                    .fontWeight(.semibold)
                    .foregroundColor(.white)
                    .padding(.horizontal, 20)
                    .padding(.vertical, 10)
                    .background(Color.black.opacity(0.6))
                    .cornerRadius(20)
                
                Button(action: isRecording ? onStopRecording : onStartRecording) {
                    HStack {
                        Image(systemName: isRecording ? "stop.circle.fill" : "record.circle")
                            .font(.title2)
                        Text(isRecording ? "Stop Recording" : "Start Recording")
                            .fontWeight(.semibold)
                    }
                    .foregroundColor(.white)
                    .padding(.horizontal, 30)
                    .padding(.vertical, 15)
                    .background(isRecording ? Color.red : Color.blue)
                    .cornerRadius(25)
                }
            }
            .padding(.bottom, 40)
        }
    }
}
