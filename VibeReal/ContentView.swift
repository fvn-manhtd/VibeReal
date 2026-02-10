import SwiftUI
import Combine
import AVFoundation

struct ConversationItem: Identifiable, Equatable {
    let id: UUID
    var text: String
    let isUser: Bool
    let timestamp: Date
}

struct WhisperCppModel: Identifiable {
    let id: String
    let name: String

    var downloadURL: URL {
        URL(string: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/\(id)")!
    }
}

@MainActor
class WhisperStreamer: ObservableObject {
    @Published var conversation: [ConversationItem] = []
    @Published var currentText: String = ""
    @Published var isRunning: Bool = false
    @Published var isModelReady: Bool = false
    @Published var isModelLoading: Bool = false
    @Published var modelProgress: Float = 0
    @Published var selectedLanguage: String = "ja"
    @Published var selectedModel: String = "ggml-small.bin"

    let availableModels: [WhisperCppModel] = [
        WhisperCppModel(id: "ggml-tiny.bin", name: "Tiny (Ultra Fast)"),
        WhisperCppModel(id: "ggml-base.bin", name: "Base (Balanced)"),
        WhisperCppModel(id: "ggml-small.bin", name: "Small (Recommended JA)"),
        WhisperCppModel(id: "ggml-medium.bin", name: "Medium (High Accuracy JA)"),
        WhisperCppModel(id: "ggml-large-v3-turbo.bin", name: "Large v3 Turbo (Best Accuracy)")
    ]

    private let whisperEngine = WhisperCppEngine()
    private let sampleStore = AudioSampleStore(maxSeconds: 20, sampleRate: WhisperCppEngine.sampleRate)
    private let audioCapture = LiveAudioCapture(sampleRate: WhisperCppEngine.sampleRate)

    private var streamingLoopTask: Task<Void, Never>?
    private var macRecorderTask: Task<Void, Never>?
    private var macRecorder: AVAudioRecorder?
    private var macRecordingURL: URL?
    private var silenceTimer: Timer?
    private var isInferenceInFlight = false
    private var currentBubbleFinalized = true
    private var currentBubbleText = ""

    private let silenceThreshold: TimeInterval = 1.2
    private let inferenceIntervalNanoseconds: UInt64 = 300_000_000
    private let inferenceWindowSeconds: TimeInterval = 4.0
    private let minInferenceAudioSeconds: TimeInterval = 0.8
    private let speechRmsThreshold: Float = 0.003

    init() {
        setupWhisper()
    }

    func setupWhisper() {
        Task {
            await loadSelectedModel()
        }
    }

    func changeModel(to modelId: String) {
        guard modelId != selectedModel else { return }

        if isRunning {
            stop()
        }

        selectedModel = modelId
        isModelReady = false
        conversation.append(
            ConversationItem(
                id: UUID(),
                text: "Loading model: \(modelId)",
                isUser: false,
                timestamp: Date()
            )
        )

        Task {
            await loadSelectedModel()
        }
    }

    func clearConversation() {
        if isRunning {
            stop()
        }

        conversation.removeAll()
        currentText = ""
        currentBubbleFinalized = true
        currentBubbleText = ""
    }

    func toggleStreaming() {
        if isRunning {
            stop()
        } else {
            start()
        }
    }

    private func loadSelectedModel() async {
        do {
            isModelLoading = true
            isModelReady = false
            modelProgress = 0

            let modelURL = try await ensureModelAvailable(for: selectedModel)
            try whisperEngine.loadModel(at: modelURL.path)

            isModelLoading = false
            isModelReady = true
            modelProgress = 1

            let modelName = availableModels.first(where: { $0.id == selectedModel })?.name ?? selectedModel
            conversation.append(
                ConversationItem(
                    id: UUID(),
                    text: "Model ready: \(modelName). Real-time transcription is configured for Japanese by default.",
                    isUser: false,
                    timestamp: Date()
                )
            )
        } catch {
            isModelLoading = false
            isModelReady = false
            conversation.append(
                ConversationItem(
                    id: UUID(),
                    text: "Model load failed: \(error.localizedDescription)",
                    isUser: false,
                    timestamp: Date()
                )
            )
            print("❌ whisper.cpp setup error: \(error)")
        }
    }

    private func ensureModelAvailable(for modelId: String) async throws -> URL {
        let destinationURL = localModelURL(for: modelId)

        if FileManager.default.fileExists(atPath: destinationURL.path) {
            return destinationURL
        }

        guard let model = availableModels.first(where: { $0.id == modelId }) else {
            throw WhisperCppError.modelLoadFailed(path: modelId)
        }

        let modelsDirectory = modelsDirectoryURL()
        try FileManager.default.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)

        let (tempFileURL, _) = try await URLSession.shared.download(from: model.downloadURL)

        if FileManager.default.fileExists(atPath: destinationURL.path) {
            try FileManager.default.removeItem(at: destinationURL)
        }

        try FileManager.default.moveItem(at: tempFileURL, to: destinationURL)
        return destinationURL
    }

    private func modelsDirectoryURL() -> URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return appSupport.appendingPathComponent("WhisperCppModels", isDirectory: true)
    }

    private func localModelURL(for modelId: String) -> URL {
        modelsDirectoryURL().appendingPathComponent(modelId)
    }

    private func start() {
        guard isModelReady else {
            return
        }

        Task {
            let granted = await requestMicrophonePermission()
            if !granted {
                await MainActor.run {
                    self.conversation.append(
                        ConversationItem(
                            id: UUID(),
                            text: "Microphone permission denied.",
                            isUser: false,
                            timestamp: Date()
                        )
                    )
                    self.isRunning = false
                }
                return
            }

            do {
                try configureAudioSessionForStreaming()

                sampleStore.clear()
                currentBubbleFinalized = true
                currentText = ""
                currentBubbleText = ""
                isInferenceInFlight = false

                if ProcessInfo.processInfo.isiOSAppOnMac {
                    try startMacFallbackStreaming()
                    isRunning = true
                    conversation.append(
                        ConversationItem(
                            id: UUID(),
                            text: "Using My Mac microphone fallback pipeline.",
                            isUser: false,
                            timestamp: Date()
                        )
                    )
                    return
                }

                audioCapture.onSamples = { [weak self] chunk in
                    self?.sampleStore.append(chunk)
                }

                try audioCapture.start()
                isRunning = true

                startStreamingLoop()
            } catch {
                await MainActor.run {
                    self.conversation.append(
                        ConversationItem(
                            id: UUID(),
                            text: "Failed to start live capture: \(error.localizedDescription)",
                            isUser: false,
                            timestamp: Date()
                        )
                    )
                    self.isRunning = false
                }
            }
        }
    }

    private func configureAudioSessionForStreaming() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker, .allowBluetooth])
        try session.setPreferredSampleRate(Double(WhisperCppEngine.sampleRate))
        try session.setPreferredIOBufferDuration(0.01)
        try session.setActive(true)
    }

    private func startStreamingLoop() {
        streamingLoopTask?.cancel()
        streamingLoopTask = Task { [weak self] in
            guard let self else { return }

            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: inferenceIntervalNanoseconds)
                await transcribeCurrentWindowIfNeeded()
            }
        }
    }

    private func startMacFallbackStreaming() throws {
        let fileURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("vibereal-live-\(UUID().uuidString).caf")

        let recorderSettings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: Double(WhisperCppEngine.sampleRate),
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsBigEndianKey: false
        ]

        let recorder = try AVAudioRecorder(url: fileURL, settings: recorderSettings)
        recorder.prepareToRecord()
        guard recorder.record() else {
            throw NSError(domain: "VibeReal.Audio", code: -1, userInfo: [NSLocalizedDescriptionKey: "Unable to start My Mac microphone recording."])
        }

        macRecorder = recorder
        macRecordingURL = fileURL

        macRecorderTask?.cancel()
        macRecorderTask = Task { [weak self] in
            guard let self else { return }

            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: inferenceIntervalNanoseconds)

                guard isRunning, let recordingURL = macRecordingURL else { continue }
                let samples = await loadRecorderSamples(url: recordingURL, maxSeconds: inferenceWindowSeconds)
                guard samples.count >= Int(minInferenceAudioSeconds * Double(WhisperCppEngine.sampleRate)) else {
                    continue
                }

                do {
                    let transcript = try await whisperEngine.transcribe(samples: samples, language: selectedLanguage)
                    if !transcript.isEmpty {
                        handleTranscriptionUpdate(text: transcript)
                    }
                } catch {
                    print("❌ whisper.cpp My Mac fallback error: \(error)")
                }
            }
        }
    }

    private func loadRecorderSamples(url: URL, maxSeconds: TimeInterval) async -> [Float] {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .utility).async {
                continuation.resume(returning: Self.readRecentSamples(from: url, maxSeconds: maxSeconds))
            }
        }
    }

    private static func readRecentSamples(from url: URL, maxSeconds: TimeInterval) -> [Float] {
        do {
            let file = try AVAudioFile(forReading: url)
            let frameCount = AVAudioFrameCount(file.length)
            guard frameCount > 0 else { return [] }

            guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: frameCount) else {
                return []
            }
            try file.read(into: buffer)

            let frameLength = Int(buffer.frameLength)
            guard frameLength > 0 else { return [] }

            var samples: [Float] = []
            if let floatChannels = buffer.floatChannelData {
                let channel0 = floatChannels[0]
                samples = Array(UnsafeBufferPointer(start: channel0, count: frameLength))
            } else if let int16Channels = buffer.int16ChannelData {
                let channel0 = int16Channels[0]
                let ints = UnsafeBufferPointer(start: channel0, count: frameLength)
                samples = ints.map { Float($0) / Float(Int16.max) }
            } else {
                return []
            }

            let maxSamples = Int(maxSeconds * Double(WhisperCppEngine.sampleRate))
            if samples.count > maxSamples {
                samples.removeFirst(samples.count - maxSamples)
            }
            return samples
        } catch {
            return []
        }
    }

    private func transcribeCurrentWindowIfNeeded() async {
        if !isRunning || isInferenceInFlight {
            return
        }

        let window = sampleStore.latest(seconds: inferenceWindowSeconds, sampleRate: WhisperCppEngine.sampleRate)
        guard window.count >= Int(minInferenceAudioSeconds * Double(WhisperCppEngine.sampleRate)) else {
            return
        }

        let rms = sampleStore.rms(seconds: 0.25, sampleRate: WhisperCppEngine.sampleRate)
        if rms < speechRmsThreshold {
            startSilenceTimerIfNeeded()
            return
        }

        silenceTimer?.invalidate()
        isInferenceInFlight = true

        do {
            let transcript = try await whisperEngine.transcribe(samples: window, language: selectedLanguage)
            if !transcript.isEmpty {
                handleTranscriptionUpdate(text: transcript)
            }
        } catch {
            print("❌ whisper.cpp streaming error: \(error)")
        }

        isInferenceInFlight = false
    }

    private func handleTranscriptionUpdate(text: String) {
        let cleaned = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty else { return }

        let mergedText = mergeStreamingText(existing: currentBubbleText, incoming: cleaned)
        currentBubbleText = mergedText
        currentText = mergedText

        if let lastIndex = conversation.indices.last,
           conversation[lastIndex].isUser,
           !currentBubbleFinalized {
            conversation[lastIndex].text = mergedText
        } else {
            startNewUserBubble()
            currentBubbleFinalized = false
            if let lastIndex = conversation.indices.last {
                conversation[lastIndex].text = mergedText
            }
        }

        silenceTimer?.invalidate()
        silenceTimer = Timer.scheduledTimer(withTimeInterval: silenceThreshold, repeats: false) { [weak self] _ in
            Task { @MainActor in
                self?.finalizeCurrentBubble()
            }
        }
    }

    private func startSilenceTimerIfNeeded() {
        guard !currentBubbleFinalized else { return }
        guard silenceTimer == nil else { return }

        silenceTimer = Timer.scheduledTimer(withTimeInterval: silenceThreshold, repeats: false) { [weak self] _ in
            Task { @MainActor in
                self?.finalizeCurrentBubble()
            }
        }
    }

    private func startNewUserBubble() {
        conversation.append(
            ConversationItem(
                id: UUID(),
                text: "",
                isUser: true,
                timestamp: Date()
            )
        )
    }

    private func finalizeCurrentBubble() {
        silenceTimer?.invalidate()
        silenceTimer = nil
        currentBubbleFinalized = true
        currentText = ""
        currentBubbleText = ""
    }

    private func requestMicrophonePermission() async -> Bool {
        await withCheckedContinuation { continuation in
            AVAudioSession.sharedInstance().requestRecordPermission { granted in
                continuation.resume(returning: granted)
            }
        }
    }

    private func stop() {
        isRunning = false
        streamingLoopTask?.cancel()
        streamingLoopTask = nil
        macRecorderTask?.cancel()
        macRecorderTask = nil

        silenceTimer?.invalidate()
        silenceTimer = nil

        audioCapture.stop()
        macRecorder?.stop()
        macRecorder = nil
        if let recordingURL = macRecordingURL {
            try? FileManager.default.removeItem(at: recordingURL)
        }
        macRecordingURL = nil
        sampleStore.clear()
        currentBubbleFinalized = true
        currentText = ""
        currentBubbleText = ""

        Task {
            try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        }
    }

    private func mergeStreamingText(existing: String, incoming: String) -> String {
        if existing.isEmpty {
            return incoming
        }

        if incoming == existing {
            return existing
        }

        if incoming.hasPrefix(existing) {
            return incoming
        }

        if existing.hasSuffix(incoming) {
            return existing
        }

        if incoming.contains(existing) {
            return incoming
        }

        if existing.contains(incoming) {
            return existing
        }

        let overlap = bestOverlapLength(suffixSource: existing, prefixSource: incoming)
        if overlap > 0 {
            let appendStart = incoming.index(incoming.startIndex, offsetBy: overlap)
            return (existing + String(incoming[appendStart...])).trimmingCharacters(in: .whitespacesAndNewlines)
        }

        return (existing + " " + incoming).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func bestOverlapLength(suffixSource: String, prefixSource: String) -> Int {
        let left = Array(suffixSource)
        let right = Array(prefixSource)
        let maxLength = min(left.count, right.count)
        guard maxLength > 0 else { return 0 }

        for length in stride(from: maxLength, through: 1, by: -1) {
            let leftSlice = left[(left.count - length)..<left.count]
            let rightSlice = right[0..<length]
            if leftSlice.elementsEqual(rightSlice) {
                return length
            }
        }

        return 0
    }
}

struct ContentView: View {
    @StateObject private var streamer = WhisperStreamer()

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
            Color(red: 0.12, green: 0.12, blue: 0.12)
                .ignoresSafeArea()

            VStack(spacing: 0) {
                HStack {
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
                            Text(languages.first(where: { $0.1 == streamer.selectedLanguage })?.0 ?? "Japanese")
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

                    Menu {
                        ForEach(streamer.availableModels) { model in
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
                            Text(streamer.availableModels.first(where: { $0.id == streamer.selectedModel })?.name ?? "Small")
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

                    if !streamer.conversation.isEmpty {
                        Button(action: { streamer.clearConversation() }) {
                            Image(systemName: "trash")
                                .font(.system(size: 14))
                                .foregroundColor(.red.opacity(0.8))
                                .padding(8)
                                .background(Color(white: 0.2))
                                .cornerRadius(8)
                        }
                        .disabled(streamer.isRunning)
                    }

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
