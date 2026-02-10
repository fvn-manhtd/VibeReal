import SwiftUI
import WhisperKit
import Combine
import AVFoundation

@MainActor
class WhisperStreamer: ObservableObject {
    @Published var text: String = "Sẵn sàng..."
    @Published var isRunning: Bool = false
    @Published var modelProgress: Float = 0
    
    private var whisperKit: WhisperKit?
    private var audioStreamTranscriber: AudioStreamTranscriber?
    
    init() {
        setupWhisper()
    }
    
    func setupWhisper() {
        Task {
            do {
                // Tự động chọn model tốt nhất cho thiết bị (thường là base hoặc tiny)
                let config = WhisperKitConfig(model: "large-v3")
                whisperKit = try await WhisperKit(config)
                text = "Model đã sẵn sàng!"
            } catch {
                text = "Lỗi tải model: \(error.localizedDescription)"
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
    
    private func start() {
        guard let whisperKit = whisperKit else { return }
        isRunning = true
        text = "Đang nghe..."
        
        Task {
            do {
                if audioStreamTranscriber == nil {
                    guard let tokenizer = whisperKit.tokenizer else {
                        text = "Lỗi: Tokenizer chưa sẵn sàng"
                        isRunning = false
                        return
                    }
                    
                    let decodingOptions = DecodingOptions()
                    
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
                self.text = "Lỗi: \(error.localizedDescription)"
                self.isRunning = false
            }
        }
    }
    
    private func stop() {
        Task {
            await audioStreamTranscriber?.stopStreamTranscription()
            await MainActor.run {
                isRunning = false
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
                .background(streamer.isRunning ? Color.red : Color.blue)
                .foregroundColor(.white)
                .cornerRadius(30)
            }
        }
        .padding()
    }
}
