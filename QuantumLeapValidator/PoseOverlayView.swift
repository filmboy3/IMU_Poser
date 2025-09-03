import UIKit
import Vision

class PoseOverlayView: UIView {
    private var bodyPoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint] = [:]
    private var lastObservationTime: TimeInterval = 0
    
    // Pose drawing configuration
    private let jointRadius: CGFloat = 6.0
    private let lineWidth: CGFloat = 3.0
    private let confidenceThreshold: Float = 0.3
    
    // Colors for different body parts
    private let headColor = UIColor.systemYellow
    private let torsoColor = UIColor.systemBlue
    private let leftArmColor = UIColor.systemGreen
    private let rightArmColor = UIColor.systemRed
    private let leftLegColor = UIColor.systemPurple
    private let rightLegColor = UIColor.systemOrange
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .clear
        isOpaque = false
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        backgroundColor = .clear
        isOpaque = false
    }
    
    override func draw(_ rect: CGRect) {
        super.draw(rect)
        
        guard let context = UIGraphicsGetCurrentContext() else { return }
        
        // Clear the context
        context.clear(rect)
        
        // Draw pose skeleton if we have valid points
        if !bodyPoints.isEmpty {
            drawPoseSkeleton(in: context)
        }
    }
    
    private func drawPoseSkeleton(in context: CGContext) {
        // Define skeletal connections
        let connections: [(VNHumanBodyPoseObservation.JointName, VNHumanBodyPoseObservation.JointName, UIColor)] = [
            // Head and neck
            (.nose, .neck, headColor),
            (.leftEye, .nose, headColor),
            (.rightEye, .nose, headColor),
            (.leftEar, .leftEye, headColor),
            (.rightEar, .rightEye, headColor),
            
            // Torso
            (.neck, .leftShoulder, torsoColor),
            (.neck, .rightShoulder, torsoColor),
            (.leftShoulder, .rightShoulder, torsoColor),
            (.leftShoulder, .leftHip, torsoColor),
            (.rightShoulder, .rightHip, torsoColor),
            (.leftHip, .rightHip, torsoColor),
            
            // Left arm
            (.leftShoulder, .leftElbow, leftArmColor),
            (.leftElbow, .leftWrist, leftArmColor),
            
            // Right arm
            (.rightShoulder, .rightElbow, rightArmColor),
            (.rightElbow, .rightWrist, rightArmColor),
            
            // Left leg
            (.leftHip, .leftKnee, leftLegColor),
            (.leftKnee, .leftAnkle, leftLegColor),
            
            // Right leg
            (.rightHip, .rightKnee, rightLegColor),
            (.rightKnee, .rightAnkle, rightLegColor)
        ]
        
        // Draw connections (bones)
        for (startJoint, endJoint, color) in connections {
            drawConnection(
                from: startJoint,
                to: endJoint,
                color: color,
                in: context
            )
        }
        
        // Draw joints (points)
        for (joint, point) in bodyPoints {
            guard point.confidence > confidenceThreshold else { continue }
            
            let color = colorForJoint(joint)
            drawJoint(point: point, color: color, in: context)
        }
    }
    
    private func drawConnection(
        from startJoint: VNHumanBodyPoseObservation.JointName,
        to endJoint: VNHumanBodyPoseObservation.JointName,
        color: UIColor,
        in context: CGContext
    ) {
        guard let startPoint = bodyPoints[startJoint],
              let endPoint = bodyPoints[endJoint],
              startPoint.confidence > confidenceThreshold,
              endPoint.confidence > confidenceThreshold else {
            return
        }
        
        let startCGPoint = convertVisionPointToViewPoint(startPoint)
        let endCGPoint = convertVisionPointToViewPoint(endPoint)
        
        context.setStrokeColor(color.cgColor)
        context.setLineWidth(lineWidth)
        context.setLineCap(.round)
        
        context.move(to: startCGPoint)
        context.addLine(to: endCGPoint)
        context.strokePath()
    }
    
    private func drawJoint(
        point: VNRecognizedPoint,
        color: UIColor,
        in context: CGContext
    ) {
        let cgPoint = convertVisionPointToViewPoint(point)
        
        // Draw outer circle (border)
        context.setFillColor(UIColor.white.cgColor)
        context.fillEllipse(in: CGRect(
            x: cgPoint.x - jointRadius - 1,
            y: cgPoint.y - jointRadius - 1,
            width: (jointRadius + 1) * 2,
            height: (jointRadius + 1) * 2
        ))
        
        // Draw inner circle (joint)
        context.setFillColor(color.cgColor)
        context.fillEllipse(in: CGRect(
            x: cgPoint.x - jointRadius,
            y: cgPoint.y - jointRadius,
            width: jointRadius * 2,
            height: jointRadius * 2
        ))
    }
    
    private func convertVisionPointToViewPoint(_ point: VNRecognizedPoint) -> CGPoint {
        // Vision coordinates are normalized (0,0) at bottom-left
        // UIView coordinates have (0,0) at top-left
        let x = point.location.x * bounds.width
        let y = (1.0 - point.location.y) * bounds.height
        return CGPoint(x: x, y: y)
    }
    
    private func colorForJoint(_ joint: VNHumanBodyPoseObservation.JointName) -> UIColor {
        switch joint {
        case .nose, .leftEye, .rightEye, .leftEar, .rightEar:
            return headColor
        case .neck:
            return torsoColor
        case .leftShoulder, .leftElbow, .leftWrist:
            return leftArmColor
        case .rightShoulder, .rightElbow, .rightWrist:
            return rightArmColor
        case .leftHip, .leftKnee, .leftAnkle:
            return leftLegColor
        case .rightHip, .rightKnee, .rightAnkle:
            return rightLegColor
        default:
            return UIColor.white
        }
    }
    
    public func processObservations(_ observations: [VNHumanBodyPoseObservation]) {
        let currentTime = CACurrentMediaTime()
        
        guard let observation = observations.first else {
            // Clear pose if no person detected
            bodyPoints.removeAll()
            DispatchQueue.main.async {
                self.setNeedsDisplay()
            }
            return
        }
        
        // Extract all recognized points
        guard let recognizedPoints = try? observation.recognizedPoints(.all) else {
            return
        }
        
        // Update body points with high-confidence detections
        bodyPoints = recognizedPoints.filter { $0.value.confidence > confidenceThreshold }
        
        // Update display
        lastObservationTime = currentTime
        DispatchQueue.main.async {
            self.setNeedsDisplay()
        }
    }
    
    // Public method to get current pose data for analysis
    public func getCurrentPoseData() -> [String: [String: Any]]? {
        guard !bodyPoints.isEmpty else { return nil }
        
        var poseData: [String: [String: Any]] = [:]
        
        for (joint, point) in bodyPoints {
            let jointName = jointNameString(from: joint)
            poseData[jointName] = [
                "x": point.location.x,
                "y": point.location.y,
                "confidence": point.confidence
            ]
        }
        
        return poseData
    }
    
    private func jointNameString(from joint: VNHumanBodyPoseObservation.JointName) -> String {
        switch joint {
        case .nose: return "nose"
        case .leftEye: return "left_eye"
        case .rightEye: return "right_eye"
        case .leftEar: return "left_ear"
        case .rightEar: return "right_ear"
        case .neck: return "neck"
        case .leftShoulder: return "left_shoulder"
        case .rightShoulder: return "right_shoulder"
        case .leftElbow: return "left_elbow"
        case .rightElbow: return "right_elbow"
        case .leftWrist: return "left_wrist"
        case .rightWrist: return "right_wrist"
        case .leftHip: return "left_hip"
        case .rightHip: return "right_hip"
        case .leftKnee: return "left_knee"
        case .rightKnee: return "right_knee"
        case .leftAnkle: return "left_ankle"
        case .rightAnkle: return "right_ankle"
        default: return "unknown"
        }
    }
    
    // Method to check if a valid pose is detected
    public func hasValidPose() -> Bool {
        let requiredJoints: [VNHumanBodyPoseObservation.JointName] = [
            .neck, .leftShoulder, .rightShoulder, .leftHip, .rightHip
        ]
        
        return requiredJoints.allSatisfy { joint in
            if let point = bodyPoints[joint] {
                return point.confidence > confidenceThreshold
            }
            return false
        }
    }
    
    // Method to get pose quality score
    public func getPoseQualityScore() -> Float {
        guard !bodyPoints.isEmpty else { return 0.0 }
        
        let totalConfidence = bodyPoints.values.reduce(0) { $0 + $1.confidence }
        return totalConfidence / Float(bodyPoints.count)
    }
}
