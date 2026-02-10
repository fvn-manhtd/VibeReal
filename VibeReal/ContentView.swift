import SwiftUI
import WhisperKit
import Combine
import AVFoundation

@MainActor
class WhisperStreamer: ObservableObject {
    @Published var text: String = "Đang tải model..."
    @Published var isRunning: Bool = false
    @Published var isModelReady: Bool = false
    @Published var modelProgress: Float = 0
    
    private var whisperKit: WhisperKit?
    private var audioStreamTranscriber: AudioStreamTranscriber?
    private var audioProcessor: AudioProcessor?
    private var macRecorder: AVAudioRecorder?
    private var macRecorderTask: Task<Void, Never>?
    private var macRecordingURL: URL?
    
    init() {
        setupWhisper()
    }
    
    func setupWhisper() {
        Task {
            do {
                text = "Đang tải model... (có thể mất vài phút lần đầu)"
                // Use "base" model - small enough for iPhone, good accuracy
                let config = WhisperKitConfig(model: "base", load: true, download: true)
                whisperKit = try await WhisperKit(config)
                isModelReady = true
                text = "Model đã sẵn sàng! Nhấn nút để bắt đầu."
            } catch {
                text = "Lỗi tải model: \(error.localizedDescription)"
                print("❌ WhisperKit setup error: \(error)")
            }
        }
    }
    
    func toggleStreaming() {
        if isRunning {
            stop()
        } else {
            start()
        }
    }

    private func makeDecodingOptions() -> DecodingOptions {
        DecodingOptions(
            verbose: true,
            task: .transcribe,
            language: "ja",
            temperature: 0.0,
            usePrefillPrompt: false,
            skipSpecialTokens: true
        )
    }
    
    private func start() {
        guard let whisperKit = whisperKit else {
            text = "Model chưa sẵn sàng, vui lòng đợi..."
            return
        }
        
        Task {
            // 1. Request microphone permission
            let granted = await requestMicrophonePermission()
            if !granted {
                await MainActor.run {
                    self.text = "Quyền truy cập micro bị từ chối. Vui lòng cấp quyền trong Cài đặt."
                    self.isRunning = false
                }
                return
            }
            
            await MainActor.run {
                self.isRunning = true
                self.text = "Đang nghe..."
            }
            
            do {
                if ProcessInfo.processInfo.isiOSAppOnMac {
                    print("🖥️ Using My Mac fallback recording mode")
                    try startMacFallbackStreaming(whisperKit: whisperKit, decodingOptions: makeDecodingOptions())
                    return
                }

                if audioStreamTranscriber == nil {
                    guard let tokenizer = whisperKit.tokenizer else {
                        text = "Lỗi: Tokenizer chưa sẵn sàng"
                        isRunning = false
                        return
                    }

                    // Use a fresh processor each run to avoid stale engine/format state.
                    let freshAudioProcessor = AudioProcessor()
                    whisperKit.audioProcessor = freshAudioProcessor
                    audioProcessor = freshAudioProcessor
                    
                    let decodingOptions = makeDecodingOptions()
                    
                    print("🎤 Initializing AudioStreamTranscriber...")
                    
                    audioStreamTranscriber = AudioStreamTranscriber(
                        audioEncoder: whisperKit.audioEncoder,
                        featureExtractor: whisperKit.featureExtractor,
                        segmentSeeker: whisperKit.segmentSeeker,
                        textDecoder: whisperKit.textDecoder,
                        tokenizer: tokenizer,
                        audioProcessor: freshAudioProcessor,
                        decodingOptions: decodingOptions
                    ) { [weak self] oldState, newState in
                        guard let self = self else { return }
                        
                        // Debug print
                        // print("🔄 Stream Update: currentText='\(newState.currentText)'")
                        
                        Task { @MainActor in
                            if newState.currentText.isEmpty {
                                if !newState.confirmedSegments.isEmpty {
                                    self.text = newState.confirmedSegments.map { $0.text }.joined(separator: " ")
                                }
                            } else {
                                let confirmedText = newState.confirmedSegments.map { $0.text }.joined(separator: " ")
                                self.text = confirmedText + (confirmedText.isEmpty ? "" : " ") + newState.currentText
                            }
                        }
                    }
                }
                
                print("▶️ Calling startStreamTranscription")
                try await audioStreamTranscriber?.startStreamTranscription()
                print("✅ startStreamTranscription returned")
            } catch {
                await MainActor.run {
                    self.text = "Lỗi: \(error.localizedDescription)"
                    self.isRunning = false
                }
                print("❌ Transcription error: \(error)")
            }
        }
    }
    
    private func requestMicrophonePermission() async -> Bool {
        return await withCheckedContinuation { continuation in
            AVAudioSession.sharedInstance().requestRecordPermission { granted in
                continuation.resume(returning: granted)
            }
        }
    }

    private func startMacFallbackStreaming(whisperKit: WhisperKit, decodingOptions: DecodingOptions) throws {
        let fileURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("vibereal-live-\(UUID().uuidString).caf")

        let recorderSettings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: 16_000,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false
        ]

        let recorder = try AVAudioRecorder(url: fileURL, settings: recorderSettings)
        recorder.prepareToRecord()
        guard recorder.record() else {
            throw WhisperError.audioProcessingFailed("Không thể bắt đầu ghi âm trên My Mac")
        }

        macRecorder = recorder
        macRecordingURL = fileURL

        macRecorderTask?.cancel()
        macRecorderTask = Task.detached { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_500_000_000)
                guard let self = self else { return }

                // Read @MainActor properties safely from the background thread
                let (stillRunning, recordingURL) = await MainActor.run {
                    (self.isRunning, self.macRecordingURL)
                }
                if Task.isCancelled || !stillRunning { break }
                guard let recordingURL else { break }

                // loadAudio now runs on a background thread (not blocking UI)
                let loadedResults = await AudioProcessor.loadAudio(
                    at: [recordingURL.path],
                    channelMode: .sumChannels(nil)
                )

                guard case let .success(samples)? = loadedResults.first,
                      samples.count >= WhisperKit.sampleRate else {
                    continue
                }

                // transcribe now runs on a background thread (not blocking UI)
                do {
                    let results = try await whisperKit.transcribe(audioArray: samples, decodeOptions: decodingOptions)
                    if let latestText = results.first?.text, !latestText.isEmpty {
                        await MainActor.run {
                            if self.isRunning {
                                self.text = latestText
                            }
                        }
                    }
                } catch {
                    print("❌ My Mac fallback transcription error: \(error)")
                }
            }
        }
    }
    
    private func stop() {
        Task {
            await audioStreamTranscriber?.stopStreamTranscription()
            // Reset transcriber so next session starts fresh
            audioStreamTranscriber = nil
            audioProcessor = nil
            macRecorderTask?.cancel()
            macRecorderTask = nil
            macRecorder?.stop()
            macRecorder = nil
            if let recordingURL = macRecordingURL {
                try? FileManager.default.removeItem(at: recordingURL)
                macRecordingURL = nil
            }
            try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
            await MainActor.run {
                isRunning = false
                text = "Đã dừng. Nhấn nút để nói lại."
            }
        }
    }
}

struct ContentView: View {
    @StateObject private var streamer = WhisperStreamer()
    
    var body: some View {
        VStack(spacing: 25) {
            Image(systemName: "waveform")
                .font(.system(size: 50))
                .foregroundColor(streamer.isRunning ? .red : .blue)
                .symbolEffect(.bounce, options: .repeating, value: streamer.isRunning)
            
            ScrollView {
                Text(streamer.text)
                    .font(.title3)
                    .padding()
                    .multilineTextAlignment(.center)
            }
            .frame(maxWidth: .infinity, maxHeight: 300)
            .background(Color.secondary.opacity(0.1))
            .cornerRadius(15)
            
            Button(action: { streamer.toggleStreaming() }) {
                HStack {
                    Image(systemName: streamer.isRunning ? "stop.fill" : "mic.fill")
                    Text(streamer.isRunning ? "Dừng" : "Bắt đầu nói")
                }
                .font(.headline)
                .padding()
                .frame(width: 200)
                .background(streamer.isRunning ? Color.red : (streamer.isModelReady ? Color.blue : Color.gray))
                .foregroundColor(.white)
                .cornerRadius(30)
            }
            .disabled(!streamer.isModelReady)
        }
        .padding()
    }
}
