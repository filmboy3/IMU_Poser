import Foundation
import CoreMotion
import Combine
import QuartzCore

struct IMUData: Codable {
    let timestamp: TimeInterval
    let acceleration: CMAcceleration
    let rotationRate: CMRotationRate
    let attitude: AttitudeData?
    
    init(timestamp: TimeInterval, acceleration: CMAcceleration, rotationRate: CMRotationRate, attitude: CMAttitude? = nil) {
        self.timestamp = timestamp
        self.acceleration = acceleration
        self.rotationRate = rotationRate
        self.attitude = attitude != nil ? AttitudeData(from: attitude!) : nil
    }
}

extension CMAcceleration: Codable {
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(x, forKey: .x)
        try container.encode(y, forKey: .y)
        try container.encode(z, forKey: .z)
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let x = try container.decode(Double.self, forKey: .x)
        let y = try container.decode(Double.self, forKey: .y)
        let z = try container.decode(Double.self, forKey: .z)
        self.init(x: x, y: y, z: z)
    }
    
    private enum CodingKeys: String, CodingKey {
        case x, y, z
    }
}

extension CMRotationRate: Codable {
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(x, forKey: .x)
        try container.encode(y, forKey: .y)
        try container.encode(z, forKey: .z)
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let x = try container.decode(Double.self, forKey: .x)
        let y = try container.decode(Double.self, forKey: .y)
        let z = try container.decode(Double.self, forKey: .z)
        self.init(x: x, y: y, z: z)
    }
    
    private enum CodingKeys: String, CodingKey {
        case x, y, z
    }
}

// Custom struct to represent CMAttitude data for encoding
struct AttitudeData: Codable {
    let roll: Double
    let pitch: Double
    let yaw: Double
    let quaternionX: Double
    let quaternionY: Double
    let quaternionZ: Double
    let quaternionW: Double
    
    init(from attitude: CMAttitude) {
        self.roll = attitude.roll
        self.pitch = attitude.pitch
        self.yaw = attitude.yaw
        self.quaternionX = attitude.quaternion.x
        self.quaternionY = attitude.quaternion.y
        self.quaternionZ = attitude.quaternion.z
        self.quaternionW = attitude.quaternion.w
    }
}

class MotionManager: ObservableObject {
    static let shared = MotionManager()
    private let motionManager = CMMotionManager()
    
    private let imuDataSubject = PassthroughSubject<IMUData, Never>()
    var imuDataPublisher: AnyPublisher<IMUData, Never> {
        imuDataSubject.eraseToAnyPublisher()
    }
    
    @Published var isActive = false
    @Published var dataRate: Double = 0.0
    
    private var lastUpdateTime: TimeInterval = 0
    private var updateCount = 0
    
    private init() {
        // Configure for high-frequency updates
        guard motionManager.isDeviceMotionAvailable else {
            print("Error: Device motion is not available.")
            return
        }
        motionManager.deviceMotionUpdateInterval = 1.0 / 100.0 // 100Hz
    }
    
    func startUpdates() {
        guard !isActive else { return }
        
        // Use a background queue for performance
        let queue = OperationQueue()
        queue.name = "com.quantumleap.CoreMotionQueue"
        queue.qualityOfService = .userInitiated
        
        motionManager.startDeviceMotionUpdates(to: queue) { [weak self] (motion, error) in
            guard let self = self, let motion = motion else {
                if let error = error {
                    print("Motion update error: \(error.localizedDescription)")
                }
                return
            }
            
            let data = IMUData(
                timestamp: motion.timestamp,
                acceleration: motion.userAcceleration,
                rotationRate: motion.rotationRate,
                attitude: motion.attitude
            )
            
            // Update data rate calculation
            self.updateDataRate()
            
            // Publish data on main thread for UI updates
            DispatchQueue.main.async {
                self.imuDataSubject.send(data)
            }
        }
        
        DispatchQueue.main.async {
            self.isActive = true
        }
        
        print("MotionManager: Started IMU updates at 100Hz")
    }
    
    func stopUpdates() {
        guard isActive else { return }
        
        motionManager.stopDeviceMotionUpdates()
        
        DispatchQueue.main.async {
            self.isActive = false
            self.dataRate = 0.0
        }
        
        updateCount = 0
        lastUpdateTime = 0
        
        print("MotionManager: Stopped IMU updates")
    }
    
    private func updateDataRate() {
        updateCount += 1
        let currentTime = CACurrentMediaTime()
        
        if lastUpdateTime == 0 {
            lastUpdateTime = currentTime
            return
        }
        
        let timeDelta = currentTime - lastUpdateTime
        if timeDelta >= 1.0 { // Update rate every second
            let rate = Double(updateCount) / timeDelta
            
            DispatchQueue.main.async {
                self.dataRate = rate
            }
            
            updateCount = 0
            lastUpdateTime = currentTime
        }
    }
    
    var isDeviceMotionAvailable: Bool {
        return motionManager.isDeviceMotionAvailable
    }
}
