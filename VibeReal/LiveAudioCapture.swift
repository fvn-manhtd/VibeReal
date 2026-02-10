import Foundation
import AVFoundation

enum LiveAudioCaptureError: LocalizedError {
    case conversionSetupFailed

    var errorDescription: String? {
        switch self {
        case .conversionSetupFailed:
            return "Unable to initialize audio converter for 16kHz mono capture."
        }
    }
}

final class AudioSampleStore {
    private let lock = NSLock()
    private let maxSamples: Int
    private var samples: [Float] = []

    init(maxSeconds: TimeInterval, sampleRate: Int) {
        maxSamples = Int(maxSeconds * Double(sampleRate))
        samples.reserveCapacity(maxSamples)
    }

    func clear() {
        lock.lock()
        samples.removeAll(keepingCapacity: true)
        lock.unlock()
    }

    func append(_ newSamples: [Float]) {
        guard !newSamples.isEmpty else { return }

        lock.lock()
        samples.append(contentsOf: newSamples)
        // Trim in larger chunks to avoid shifting memory on every callback.
        if samples.count > (maxSamples * 2) {
            samples.removeFirst(samples.count - maxSamples)
        }
        lock.unlock()
    }

    func latest(seconds: TimeInterval, sampleRate: Int) -> [Float] {
        lock.lock()
        defer { lock.unlock() }

        let required = Int(seconds * Double(sampleRate))
        if required <= 0 {
            return []
        }

        return Array(samples.suffix(required))
    }

    func rms(seconds: TimeInterval, sampleRate: Int) -> Float {
        lock.lock()
        defer { lock.unlock() }

        let required = Int(seconds * Double(sampleRate))
        if required <= 0 {
            return 0
        }

        let window = samples.suffix(required)
        guard !window.isEmpty else { return 0 }

        let energy = window.reduce(Float(0)) { partial, value in
            partial + (value * value)
        }

        return sqrt(energy / Float(window.count))
    }
}

final class LiveAudioCapture {
    var onSamples: (([Float]) -> Void)?

    private let engine = AVAudioEngine()
    private let outputFormat: AVAudioFormat
    private var converter: AVAudioConverter?

    init(sampleRate: Int) {
        outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sampleRate),
            channels: 1,
            interleaved: false
        )!
    }

    func start() throws {
        let inputNode = engine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)

        guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            throw LiveAudioCaptureError.conversionSetupFailed
        }

        self.converter = converter

        inputNode.removeTap(onBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: inputFormat) { [weak self] pcmBuffer, _ in
            self?.process(buffer: pcmBuffer)
        }

        engine.prepare()
        try engine.start()
    }

    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        converter = nil
    }

    private func process(buffer: AVAudioPCMBuffer) {
        guard let converter else { return }

        let ratio = outputFormat.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 32
        guard let convertedBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: capacity) else {
            return
        }

        var hasProvidedInput = false
        let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
            if hasProvidedInput {
                outStatus.pointee = .noDataNow
                return nil
            }

            hasProvidedInput = true
            outStatus.pointee = .haveData
            return buffer
        }

        var error: NSError?
        let status = converter.convert(to: convertedBuffer, error: &error, withInputFrom: inputBlock)
        if status != .haveData && status != .inputRanDry {
            return
        }

        guard let floatChannelData = convertedBuffer.floatChannelData?[0] else {
            return
        }

        let frameLength = Int(convertedBuffer.frameLength)
        guard frameLength > 0 else { return }

        let chunk = Array(UnsafeBufferPointer(start: floatChannelData, count: frameLength))
        onSamples?(chunk)
    }
}
