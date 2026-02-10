import SwiftUI
import WhisperKit
import Combine
import AVFoundation

struct ConversationItem: Identifiable, Equatable {
    let id: UUID
    var text: String
    let isUser: Bool // true for user, false for system/other
    let timestamp: Date
}

@MainActor
class WhisperStreamer: ObservableObject {
    @Published var conversation: [ConversationItem] = []
    @Published var currentText: String = "" // Tracks ongoing speech
    @Published var isRunning: Bool = false
    @Published var isModelReady: Bool = false
    @Published var isModelLoading: Bool = false
    @Published var modelProgress: Float = 0
    @Published var selectedLanguage: String = "en" // Default English
    @Published var selectedModel: String = "base"
    
    // Available WhisperKit models (downloaded on demand)
    let availableModels: [(name: String, id: String)] = [
        ("Tiny", "tiny"),
        ("Tiny (EN)", "tiny.en"),
        ("Base", "base"),
        ("Base (EN)", "base.en"),
        ("Small", "small"),
        ("Small (EN)", "small.en"),
        ("Medium", "medium"),
        ("Large v2", "large-v2"),
        ("Large v3", "large-v3"),
        ("Distil Large v3", "distil-large-v3")
    ]
    
    private var whisperKit: WhisperKit?
    private var audioStreamTranscriber: AudioStreamTranscriber?
    private var audioProcessor: AudioProcessor?
    private var macRecorder: AVAudioRecorder?
    private var macRecorderTask: Task<Void, Never>?
    private var macRecordingURL: URL?
    
    private var silenceTimer: Timer?
    private let silenceThreshold: TimeInterval = 1.5 // Seconds to wait before finalizing a bubble
    private var displayedTextOffset: Int = 0 // Tracks how much accumulated text has been committed to previous bubbles
    private var lastFullText: String = "" // Stores the full accumulated text for diffing
    
    init() {
        setupWhisper()
    }
    
    func setupWhisper() {
        Task {
            do {
                isModelLoading = true
                let config = WhisperKitConfig(model: selectedModel, load: true, download: true)
                whisperKit = try await WhisperKit(config)
                isModelReady = true
                isModelLoading = false
                
                // Add initial system message
                conversation.append(ConversationItem(id: UUID(), text: "Model đã sẵn sàng! Nhấn nút để bắt đầu.", isUser: false, timestamp: Date()))
            } catch {
                isModelLoading = false
                conversation.append(ConversationItem(id: UUID(), text: "Lỗi tải model: \(error.localizedDescription)", isUser: false, timestamp: Date()))
                print("❌ WhisperKit setup error: \(error)")
            }
        }
    }
    
    func changeModel(to modelId: String) {
        guard modelId != selectedModel else { return }
        // Stop streaming if active
        if isRunning {
            stop()
        }
        selectedModel = modelId
        isModelReady = false
        whisperKit = nil
        audioStreamTranscriber = nil
        audioProcessor = nil
        conversation.append(ConversationItem(id: UUID(), text: "Loading model: \(modelId)...", isUser: false, timestamp: Date()))
        setupWhisper()
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
            language: selectedLanguage,
            temperature: 0.0,
            usePrefillPrompt: false,
            skipSpecialTokens: true
        )
    }
    
    private func start() {
        guard let whisperKit = whisperKit else {
            return
        }
        
        Task {
            // 1. Request microphone permission
            let granted = await requestMicrophonePermission()
            if !granted {
                await MainActor.run {
                    self.conversation.append(ConversationItem(id: UUID(), text: "Quyền truy cập micro bị từ chối.", isUser: false, timestamp: Date()))
                    self.isRunning = false
                }
                return
            }
            
            await MainActor.run {
                self.isRunning = true
                self.displayedTextOffset = 0
                self.lastFullText = ""
                // Start a new user bubble
                self.startNewUserBubble()
            }
            
            do {
                if ProcessInfo.processInfo.isiOSAppOnMac {
                    print("🖥️ Using My Mac fallback recording mode")
                    try startMacFallbackStreaming(whisperKit: whisperKit, decodingOptions: makeDecodingOptions())
                    return
                }

                if audioStreamTranscriber == nil {
                    guard let tokenizer = whisperKit.tokenizer else {
                        isRunning = false
                        return
                    }

                    // Use a fresh processor each run to avoid stale engine/format state.
                    let freshAudioProcessor = AudioProcessor()
                    whisperKit.audioProcessor = freshAudioProcessor
                    audioProcessor = freshAudioProcessor
                    
                    let decodingOptions = makeDecodingOptions()
                    
                    print("🎤 Initializing AudioStreamTranscriber with language: \(selectedLanguage)")
                    
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
                        
                        Task { @MainActor in
                            self.handleTranscriptionUpdate(newState: newState)
                        }
                    }
                }
                
                print("▶️ Calling startStreamTranscription")
                try await audioStreamTranscriber?.startStreamTranscription()
                print("✅ startStreamTranscription returned")
            } catch {
                await MainActor.run {
                    self.conversation.append(ConversationItem(id: UUID(), text: "Lỗi: \(error.localizedDescription)", isUser: false, timestamp: Date()))
                    self.isRunning = false
                }
                print("❌ Transcription error: \(error)")
            }
        }
    }
    
    private func handleTranscriptionUpdate(newState: AudioStreamTranscriber.State) {
        // Reset silence timer on any update
        silenceTimer?.invalidate()
        
        let fullText = (newState.confirmedSegments.map { $0.text }.joined(separator: " ") + " " + newState.currentText).trimmingCharacters(in: .whitespacesAndNewlines)
        lastFullText = fullText
        
        // Extract only the NEW text since last committed offset
        let visibleText: String
        if displayedTextOffset < fullText.count {
            let startIdx = fullText.index(fullText.startIndex, offsetBy: displayedTextOffset)
            visibleText = String(fullText[startIdx...]).trimmingCharacters(in: .whitespacesAndNewlines)
        } else {
            visibleText = ""
        }
        
        if !visibleText.isEmpty {
            // Update the last item if it is a user bubble
            if let lastIndex = conversation.indices.last, conversation[lastIndex].isUser {
                conversation[lastIndex].text = visibleText
            } else {
                startNewUserBubble()
                conversation[conversation.indices.last!].text = visibleText
            }
            
            // Set timer to finalize bubble if silence persists
            silenceTimer = Timer.scheduledTimer(withTimeInterval: silenceThreshold, repeats: false) { [weak self] _ in
                Task { @MainActor in
                    self?.finalizeCurrentBubbleAndPrepareNext()
                }
            }
        }
    }
    
    private func startNewUserBubble() {
        // Only start new if previous wasn't empty or if it's the very first one
        if let last = conversation.last, last.isUser, last.text.isEmpty {
            return // Re-use empty bubble
        }
        conversation.append(ConversationItem(id: UUID(), text: "", isUser: true, timestamp: Date()))
    }
    
    private func finalizeCurrentBubbleAndPrepareNext() {
        // Commit all accumulated text up to this point
        displayedTextOffset = lastFullText.count
        
        // Only start a new bubble if the current one has content
        if let last = conversation.last, last.isUser, !last.text.isEmpty {
            startNewUserBubble()
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
        // ... (Same as before, simplified for brevity in this update)
        // For the sake of this task, I will assume iOS mainly or Mac with proper mic.
        // The original code had it, I'll keep it but clean up.
        // NOTE: The original `startMacFallbackStreaming` logic was creating a loop.
        // I will re-implement it briefly or assume standard `audioStreamTranscriber` works for the user's focus (UI).
        // But to be safe, I'll copy the logic back if needed.
        // Actually, the user is on Mac (OS version: mac).
        // So `startMacFallbackStreaming` IS CRITICAL.
        
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
            var lastTextCount = 0
            
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 500_000_000) // 0.5s interval for more responsive updates
                guard let self = self else { return }

                let (stillRunning, recordingURL) = await MainActor.run {
                    (self.isRunning, self.macRecordingURL)
                }
                if Task.isCancelled || !stillRunning { break }
                guard let recordingURL else { break }

                let loadedResults = await AudioProcessor.loadAudio(
                    at: [recordingURL.path],
                    channelMode: .sumChannels(nil)
                )

                guard case let .success(samples)? = loadedResults.first,
                      samples.count >= WhisperKit.sampleRate else {
                    continue
                }

                do {
                    let results = try await whisperKit.transcribe(audioArray: samples, decodeOptions: decodingOptions)
                    if let latestText = results.first?.text, !latestText.isEmpty {
                        await MainActor.run {
                            if self.isRunning {
                                // For fallback mode, we just update the text (naive appending)
                                // Only update if text length grew
                                if latestText.count > lastTextCount {
                                    self.handleTranscriptionUpdateTextOnly(text: latestText)
                                    lastTextCount = latestText.count
                                }
                            }
                        }
                    }
                } catch {
                    print("❌ My Mac fallback transcription error: \(error)")
                }
            }
        }
    }
    
    private func handleTranscriptionUpdateTextOnly(text: String) {
        // Reset silence timer
        silenceTimer?.invalidate()
        lastFullText = text
        
        // Extract only the NEW text since last committed offset
        let visibleText: String
        if displayedTextOffset < text.count {
            let startIdx = text.index(text.startIndex, offsetBy: displayedTextOffset)
            visibleText = String(text[startIdx...]).trimmingCharacters(in: .whitespacesAndNewlines)
        } else {
            visibleText = ""
        }
        
        if !visibleText.isEmpty {
            if let lastIndex = conversation.indices.last, conversation[lastIndex].isUser {
                conversation[lastIndex].text = visibleText
            } else {
                startNewUserBubble()
                conversation[conversation.indices.last!].text = visibleText
            }
            
            silenceTimer = Timer.scheduledTimer(withTimeInterval: silenceThreshold, repeats: false) { [weak self] _ in
                Task { @MainActor in
                    self?.finalizeCurrentBubbleAndPrepareNext()
                }
            }
        }
    }
    
    private func stop() {
        Task {
            await audioStreamTranscriber?.stopStreamTranscription()
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
            }
        }
    }
}

struct ContentView: View {
    @StateObject private var streamer = WhisperStreamer()
    
    // Languages map: Display Name -> Code
    let languages = [
        ("English", "en"),
        ("Japanese", "ja"),
        ("Vietnamese", "vi"),
        ("Chinese", "zh"),
        ("Korean", "ko"),
        ("French", "fr"),
        ("German", "de"),
        ("Spanish", "es")
    ]
    
    var body: some View {
        ZStack {
            // Background
            Color(red: 0.12, green: 0.12, blue: 0.12)
                .ignoresSafeArea()
            
            VStack(spacing: 0) {
                // Header
                HStack {
                    // Language Selector
                    Menu {
                        ForEach(languages, id: \.1) { lang in
                            Button(action: {
                                streamer.selectedLanguage = lang.1
                            }) {
                                HStack {
                                    Text(lang.0)
                                    if streamer.selectedLanguage == lang.1 {
                                        Image(systemName: "checkmark")
                                    }
                                }
                            }
                        }
                    } label: {
                        HStack(spacing: 4) {
                            Text(languages.first(where: { $0.1 == streamer.selectedLanguage })?.0 ?? "English")
                                .font(.headline)
                                .foregroundColor(.white)
                            Image(systemName: "chevron.down")
                                .font(.caption)
                                .foregroundColor(.gray)
                        }
                        .padding(.vertical, 8)
                        .padding(.horizontal, 12)
                        .background(Color(white: 0.2))
                        .cornerRadius(8)
                    }
                    
                    // Model Selector
                    Menu {
                        ForEach(streamer.availableModels, id: \.id) { model in
                            Button(action: {
                                streamer.changeModel(to: model.id)
                            }) {
                                HStack {
                                    Text(model.name)
                                    if streamer.selectedModel == model.id {
                                        Image(systemName: "checkmark")
                                    }
                                }
                            }
                        }
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "cpu")
                                .font(.caption)
                                .foregroundColor(.green)
                            Text(streamer.availableModels.first(where: { $0.id == streamer.selectedModel })?.name ?? "Base")
                                .font(.headline)
                                .foregroundColor(.white)
                            if streamer.isModelLoading {
                                ProgressView()
                                    .scaleEffect(0.7)
                                    .tint(.green)
                            } else {
                                Image(systemName: "chevron.down")
                                    .font(.caption)
                                    .foregroundColor(.gray)
                            }
                        }
                        .padding(.vertical, 8)
                        .padding(.horizontal, 12)
                        .background(Color(white: 0.2))
                        .cornerRadius(8)
                    }
                    .disabled(streamer.isRunning)
                    
                    Spacer()
                    
                    if streamer.isRunning {
                        Button(action: { streamer.toggleStreaming() }) {
                            HStack(spacing: 6) {
                                Image(systemName: "square.fill")
                                    .font(.caption)
                                Text("Stop")
                                    .font(.subheadline)
                                    .bold()
                            }
                            .foregroundColor(.white)
                            .padding(.vertical, 8)
                            .padding(.horizontal, 12)
                            .background(Color.white.opacity(0.1))
                            .cornerRadius(8)
                        }
                    }
                }
                .padding()
                .background(Color(red: 0.12, green: 0.12, blue: 0.12))
                
                // Chat Area
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(spacing: 16) {
                            ForEach(streamer.conversation) { item in
                                ConversationBubble(item: item)
                                    .id(item.id)
                            }
                        }
                        .padding()
                    }
                    .onChange(of: streamer.conversation) { _ in
                        if let lastId = streamer.conversation.last?.id {
                            withAnimation {
                                proxy.scrollTo(lastId, anchor: .bottom)
                            }
                        }
                    }
                }
                
                // Footer / Controls
                if !streamer.isRunning {
                    VStack {
                        Button(action: { streamer.toggleStreaming() }) {
                            Image(systemName: "mic.fill")
                                .font(.system(size: 24))
                                .foregroundColor(.green)
                                .frame(width: 60, height: 60)
                                .background(Color(white: 0.2))
                                .clipShape(Circle())
                        }
                        .disabled(!streamer.isModelReady)
                        .opacity(streamer.isModelReady ? 1 : 0.5)
                        
                        Text(streamer.isModelReady ? "Tap to speak" : "Loading model...")
                            .font(.caption)
                            .foregroundColor(.gray)
                            .padding(.top, 8)
                    }
                    .padding(.bottom, 30)
                } else {
                    // Visualizer placeholder
                    HStack(spacing: 4) {
                        ForEach(0..<5) { _ in
                            Circle()
                                .fill(Color.green)
                                .frame(width: 8, height: 8)
                                .opacity(0.8)
                        }
                    }
                    .padding()
                }
            }
        }
        .preferredColorScheme(.dark)
    }
}

struct ConversationBubble: View {
    let item: ConversationItem
    
    var body: some View {
        HStack(alignment: .bottom) {
            if item.isUser {
                Spacer()
                VStack(alignment: .trailing, spacing: 4) {
                    Text(item.text)
                        .font(.body)
                        .foregroundColor(.white)
                        .padding(12)
                        .background(Color.green.opacity(0.8))
                        .cornerRadius(16)
                    
//                    Text(item.timestamp, style: .time)
//                        .font(.caption2)
//                        .foregroundColor(.gray)
                }
            } else {
                VStack(alignment: .leading, spacing: 4) {
                    Text(item.text)
                        .font(.body)
                        .foregroundColor(.white)
                        .padding(12)
                        .background(Color(white: 0.2))
                        .cornerRadius(16)
                }
                Spacer()
            }
        }
    }
}
