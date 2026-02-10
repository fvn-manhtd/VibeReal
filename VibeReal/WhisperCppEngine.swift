import Foundation
import whisper

nonisolated enum WhisperCppError: LocalizedError {
    case modelLoadFailed(path: String)
    case modelNotLoaded
    case transcriptionFailed(code: Int32)

    var errorDescription: String? {
        switch self {
        case .modelLoadFailed(let path):
            return "Failed to load whisper.cpp model at \(path)."
        case .modelNotLoaded:
            return "whisper.cpp model is not loaded."
        case .transcriptionFailed(let code):
            return "whisper.cpp transcription failed with code \(code)."
        }
    }
}

/// Explicitly nonisolated to escape the project-wide `SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor`.
/// This class manages its own thread safety via `contextLock` and `inferenceQueue`.
nonisolated final class WhisperCppEngine: @unchecked Sendable {
    static let sampleRate = 16_000
    private static let maxInferenceDurationSeconds: Double = 8.0

    private let inferenceQueue = DispatchQueue(label: "com.vibereal.whispercpp.inference", qos: .userInitiated)
    private let contextLock = NSLock()
    private var context: OpaquePointer?

    deinit {
        unloadModel()
    }

    func loadModel(at path: String) throws {
        print("🔧 WhisperCppEngine: Loading model from \(path)")
        var contextParams = whisper_context_default_params()
#if targetEnvironment(simulator)
        let runningAsiOSAppOnMac = false
        contextParams.use_gpu = false
#else
        let runningAsiOSAppOnMac = ProcessInfo.processInfo.isiOSAppOnMac
        contextParams.use_gpu = !runningAsiOSAppOnMac
#endif
        if runningAsiOSAppOnMac {
            print("🔧 WhisperCppEngine: forcing CPU backend for iOS app on Mac")
        }

        guard let newContext = path.withCString({ whisper_init_from_file_with_params($0, contextParams) }) else {
            throw WhisperCppError.modelLoadFailed(path: path)
        }

        contextLock.lock()
        if let oldContext = context {
            whisper_free(oldContext)
        }
        context = newContext
        contextLock.unlock()
        print("✅ WhisperCppEngine: Model loaded successfully")
    }

    func unloadModel() {
        contextLock.lock()
        if let currentContext = context {
            whisper_free(currentContext)
            context = nil
        }
        contextLock.unlock()
    }

    func transcribe(samples: [Float], language: String) async throws -> String {
        if samples.isEmpty {
            return ""
        }

        print("🎤 WhisperCppEngine: transcribe called with \(samples.count) samples, language=\(language)")

        return try await withCheckedThrowingContinuation { continuation in
            inferenceQueue.async { [self] in
                self.contextLock.lock()
                guard let context = self.context else {
                    self.contextLock.unlock()
                    print("❌ WhisperCppEngine: Model not loaded")
                    continuation.resume(throwing: WhisperCppError.modelNotLoaded)
                    return
                }
                let startedAt = DispatchTime.now().uptimeNanoseconds
                let timeoutNanoseconds = UInt64(Self.maxInferenceDurationSeconds * 1_000_000_000)
                let deadline = startedAt + timeoutNanoseconds
                let deadlineBox = Unmanaged.passRetained(DeadlineBox(deadlineNanoseconds: deadline))
                defer {
                    deadlineBox.release()
                    self.contextLock.unlock()
                }

                var params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY)
                params.print_realtime = false
                params.print_progress = false
                params.print_timestamps = false
                params.print_special = false
                params.translate = false
                params.no_context = true
                params.no_timestamps = true
                params.single_segment = true   // Critical for real-time: forces single segment output
                params.temperature = 0
                params.temperature_inc = 0     // Disable temperature fallback (prevents slow retries & hallucinations)
                params.suppress_blank = true   // Suppress blank/silence tokens
                params.suppress_non_speech_tokens = true
                params.max_tokens = 64
                params.n_threads = Int32(max(4, min(8, ProcessInfo.processInfo.activeProcessorCount)))
                let maxAudioCtx = max(1, whisper_n_audio_ctx(context))
                params.audio_ctx = Int32(min(maxAudioCtx, 128))
                let sampleDurationMs = Int32(sampleBufferDurationMs(samples.count))
                params.duration_ms = max(1000, min(2000, sampleDurationMs))
                params.abort_callback_user_data = deadlineBox.toOpaque()
                params.abort_callback = { userData in
                    guard let userData else { return false }
                    let deadlineHolder = Unmanaged<DeadlineBox>.fromOpaque(userData).takeUnretainedValue()
                    return DispatchTime.now().uptimeNanoseconds >= deadlineHolder.deadlineNanoseconds
                }
                print("🎛️ Whisper params: threads=\(params.n_threads), audio_ctx=\(params.audio_ctx), duration_ms=\(params.duration_ms)")

                // Keep the language string alive for the entire duration of whisper_full
                let languageCopy = language
                let resultCode: Int32 = languageCopy.withCString { languagePtr in
                    params.language = languagePtr
                    return samples.withUnsafeBufferPointer { sampleBuffer in
                        guard let baseAddress = sampleBuffer.baseAddress else { return -1 }
                        return whisper_full(context, params, baseAddress, Int32(sampleBuffer.count))
                    }
                }
                let elapsedMs = Double(DispatchTime.now().uptimeNanoseconds - startedAt) / 1_000_000

                guard resultCode == 0 else {
                    if DispatchTime.now().uptimeNanoseconds >= deadline {
                        print("⏱️ WhisperCppEngine: inference timed out after \(String(format: "%.0f", elapsedMs))ms")
                        continuation.resume(returning: "")
                    } else {
                        print("❌ WhisperCppEngine: whisper_full failed with code \(resultCode)")
                        continuation.resume(throwing: WhisperCppError.transcriptionFailed(code: resultCode))
                    }
                    return
                }

                let segmentCount = Int(whisper_full_n_segments(context))
                if segmentCount == 0 {
                    print("⚠️ WhisperCppEngine: 0 segments returned (\(String(format: "%.0f", elapsedMs))ms)")
                    continuation.resume(returning: "")
                    return
                }

                let text = (0..<segmentCount)
                    .compactMap { index -> String? in
                        guard let cString = whisper_full_get_segment_text(context, Int32(index)) else {
                            return nil
                        }
                        return String(cString: cString)
                    }
                    .joined(separator: " ")
                    .trimmingCharacters(in: .whitespacesAndNewlines)

                print("🎤 WhisperCppEngine: transcription result (\(segmentCount) segments, \(String(format: "%.0f", elapsedMs))ms): \(text.prefix(100))")
                continuation.resume(returning: text)
            }
        }
    }
}

nonisolated private final class DeadlineBox: @unchecked Sendable {
    let deadlineNanoseconds: UInt64

    init(deadlineNanoseconds: UInt64) {
        self.deadlineNanoseconds = deadlineNanoseconds
    }
}

@inline(__always)
nonisolated private func sampleBufferDurationMs(_ sampleCount: Int) -> Int {
    guard sampleCount > 0 else { return 0 }
    return (sampleCount * 1000) / WhisperCppEngine.sampleRate
}
