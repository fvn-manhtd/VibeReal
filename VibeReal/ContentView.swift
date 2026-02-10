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
    
    private func configureAudioSession() throws {
        let audioSession = AVAudioSession.sharedInstance()
        try audioSession.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker, .allowBluetooth])
        try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        print("✅ Audio session configured successfully")
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
            
            // 2. Configure audio session
            do {
                try configureAudioSession()
            } catch {
                await MainActor.run {
                    self.text = "Lỗi cấu hình audio: \(error.localizedDescription)"
                    self.isRunning = false
                }
                print("❌ Audio session error: \(error)")
                return
            }
            
            await MainActor.run {
                self.isRunning = true
                self.text = "Đang nghe..."
            }
            
            do {
                if audioStreamTranscriber == nil {
                    guard let tokenizer = whisperKit.tokenizer else {
                        text = "Lỗi: Tokenizer chưa sẵn sàng"
                        isRunning = false
                        return
                    }
                    
                    let decodingOptions = DecodingOptions(language: "vi")
                    
                    audioStreamTranscriber = AudioStreamTranscriber(
                        audioEncoder: whisperKit.audioEncoder,
                        featureExtractor: whisperKit.featureExtractor,
                        segmentSeeker: whisperKit.segmentSeeker,
                        textDecoder: whisperKit.textDecoder,
                        tokenizer: tokenizer,
                        audioProcessor: whisperKit.audioProcessor,
                        decodingOptions: decodingOptions
                    ) { [weak self] oldState, newState in
                        guard let self = self else { return }
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
                
                try await audioStreamTranscriber?.startStreamTranscription()
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
    
    private func stop() {
        Task {
            await audioStreamTranscriber?.stopStreamTranscription()
            // Reset transcriber so next session starts fresh
            audioStreamTranscriber = nil
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
