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
        contextParams.use_gpu = false
#else
        contextParams.use_gpu = true
#endif

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

                var params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY)
                params.print_realtime = false
                params.print_progress = false
                params.print_timestamps = false
                params.print_special = false
                params.translate = false
                params.no_context = true
                params.no_timestamps = true
                params.single_segment = false
                params.temperature = 0
                params.n_threads = Int32(max(1, min(8, ProcessInfo.processInfo.activeProcessorCount - 1)))

                // Keep the language string alive for the entire duration of whisper_full
                let languageCopy = language
                let resultCode: Int32 = languageCopy.withCString { languagePtr in
                    params.language = languagePtr
                    return samples.withUnsafeBufferPointer { sampleBuffer in
                        guard let baseAddress = sampleBuffer.baseAddress else { return -1 }
                        return whisper_full(context, params, baseAddress, Int32(sampleBuffer.count))
                    }
                }

                guard resultCode == 0 else {
                    self.contextLock.unlock()
                    print("❌ WhisperCppEngine: whisper_full failed with code \(resultCode)")
                    continuation.resume(throwing: WhisperCppError.transcriptionFailed(code: resultCode))
                    return
                }

                let segmentCount = Int(whisper_full_n_segments(context))
                if segmentCount == 0 {
                    self.contextLock.unlock()
                    print("⚠️ WhisperCppEngine: 0 segments returned")
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

                self.contextLock.unlock()
                print("🎤 WhisperCppEngine: transcription result (\(segmentCount) segments): \(text.prefix(100))")
                continuation.resume(returning: text)
            }
        }
    }
}
