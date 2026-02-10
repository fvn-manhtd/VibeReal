import SwiftUI
import Combine
import AVFoundation

nonisolated struct ConversationItem: Identifiable, Equatable, Sendable {
    let id: UUID
    var text: String
    let isUser: Bool
    let timestamp: Date
}

nonisolated struct WhisperCppModel: Identifiable, Sendable {
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
    private let inferenceIntervalNanoseconds: UInt64 = 200_000_000
    private let inferenceWindowSeconds: TimeInterval = 1.6
    private let minInferenceAudioSeconds: TimeInterval = 1.1
    private let speechRmsThreshold: Float = 0.0001
    private var consecutiveSilenceCount: Int = 0
    private var streamingStartTime: Date?

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
                sampleStore.clear()
                currentBubbleFinalized = true
                currentText = ""
                currentBubbleText = ""
                isInferenceInFlight = false
                streamingStartTime = Date()

                let onMac = ProcessInfo.processInfo.isiOSAppOnMac
                print("🎤 WhisperStreamer: isiOSAppOnMac = \(onMac)")

                // Always try to configure audio session (it IS available for iOS apps on Mac)
                configureAudioSessionForStreaming()

                // Try AVAudioEngine first on ALL platforms (including Mac)
                var useAudioEngine = true
                if onMac {
                    // On Mac, AVAudioEngine may fail — try it, fall back to AVAudioRecorder
                    do {
                        audioCapture.onSamples = { [weak self] chunk in
                            self?.sampleStore.append(chunk)
                        }
                        try audioCapture.start()
                        print("✅ AVAudioEngine works on Mac!")
                    } catch {
                        print("⚠️ AVAudioEngine failed on Mac: \(error). Falling back to AVAudioRecorder.")
                        useAudioEngine = false
                    }
                }

                if onMac && !useAudioEngine {
                    // Fallback: use AVAudioRecorder file-based pipeline
                    try startMacFallbackStreaming()
                    isRunning = true
                    conversation.append(
                        ConversationItem(
                            id: UUID(),
                            text: "🎙️ Listening via Mac microphone...",
                            isUser: false,
                            timestamp: Date()
                        )
                    )
                    return
                }

                if !onMac {
                    // iOS path: start AVAudioEngine normally
                    audioCapture.onSamples = { [weak self] chunk in
                        self?.sampleStore.append(chunk)
                    }
                    try audioCapture.start()
                }

                isRunning = true
                startStreamingLoop()

                conversation.append(
                    ConversationItem(
                        id: UUID(),
                        text: "🎙️ Listening...",
                        isUser: false,
                        timestamp: Date()
                    )
                )
            } catch {
                print("❌ WhisperStreamer: Failed to start capture: \(error)")
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

    private func configureAudioSessionForStreaming() {
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker, .allowBluetooth])
            try session.setPreferredSampleRate(Double(WhisperCppEngine.sampleRate))
            try session.setPreferredIOBufferDuration(0.01)
            try session.setActive(true)
            print("✅ Audio session configured successfully")
        } catch {
            // On Mac, audio session config may partially fail — this is OK, continue anyway
            print("⚠️ Audio session configuration error (may be OK on Mac): \(error)")
        }
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
            .appendingPathComponent("vibereal-live-\(UUID().uuidString).wav")

        let recorderSettings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: Double(WhisperCppEngine.sampleRate),
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false
        ]

        print("🎙️ Mac fallback: recording to \(fileURL.path)")

        let recorder = try AVAudioRecorder(url: fileURL, settings: recorderSettings)
        recorder.isMeteringEnabled = true
        recorder.prepareToRecord()
        guard recorder.record() else {
            throw NSError(domain: "VibeReal.Audio", code: -1, userInfo: [NSLocalizedDescriptionKey: "Unable to start My Mac microphone recording."])
        }

        print("✅ Mac fallback: AVAudioRecorder started, isRecording=\(recorder.isRecording)")

        macRecorder = recorder
        macRecordingURL = fileURL

        macRecorderTask?.cancel()
        macRecorderTask = Task { [weak self] in
            guard let self else { return }

            // Wait a moment for recorder to accumulate initial audio
            try? await Task.sleep(nanoseconds: 500_000_000)

            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: inferenceIntervalNanoseconds)

                guard isRunning, let recordingURL = macRecordingURL else { continue }

                // Force-flush the recorder's buffer by updating meters
                macRecorder?.updateMeters()

                let samples = await loadRecorderSamples(url: recordingURL, maxSeconds: inferenceWindowSeconds)

                print("🎙️ Mac fallback: read \(samples.count) samples from recorder file")

                guard samples.count >= Int(minInferenceAudioSeconds * Double(WhisperCppEngine.sampleRate)) else {
                    print("⚠️ Mac fallback: not enough samples (\(samples.count) < \(Int(minInferenceAudioSeconds * Double(WhisperCppEngine.sampleRate))))")
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
        let sampleCount = window.count
        let minRequired = Int(minInferenceAudioSeconds * Double(WhisperCppEngine.sampleRate))
        guard sampleCount >= minRequired else {
            print("📊 Skipping inference: only \(sampleCount)/\(minRequired) samples")
            return
        }

        let rms = sampleStore.rms(seconds: 0.3, sampleRate: WhisperCppEngine.sampleRate)
        print("📊 Audio stats: samples=\(sampleCount), RMS=\(String(format: "%.6f", rms)), silenceCount=\(consecutiveSilenceCount)")

        // Grace period: skip silence detection for first 3 seconds to allow warmup
        let secondsSinceStart = Date().timeIntervalSince(streamingStartTime ?? Date())
        let pastWarmup = secondsSinceStart > 3.0

        // Only skip transcription after many consecutive silence detections AND past warmup
        if rms < speechRmsThreshold && pastWarmup {
            consecutiveSilenceCount += 1
            if consecutiveSilenceCount > 8 {
                startSilenceTimerIfNeeded()
                return
            }
        } else {
            consecutiveSilenceCount = 0
        }

        silenceTimer?.invalidate()
        isInferenceInFlight = true
        defer { isInferenceInFlight = false }

        // Pad audio with silence to meet whisper.cpp's internal minimum of 1 second (16000 samples)
        var samples = window
        let whisperMinSamples = WhisperCppEngine.sampleRate  // 16000 = 1 second
        if samples.count < whisperMinSamples {
            samples.append(contentsOf: [Float](repeating: 0, count: whisperMinSamples - samples.count))
        }

        do {
            let transcript = try await whisperEngine.transcribe(samples: samples, language: selectedLanguage)
            if !transcript.isEmpty {
                print("✅ Got transcript: \(transcript.prefix(80))")
                handleTranscriptionUpdate(text: transcript)
            } else {
                print("📊 Whisper returned empty transcript")
            }
        } catch {
            print("❌ whisper.cpp streaming error: \(error)")
        }
    }

    /// Known whisper hallucination patterns to filter out
    private static let hallucinationPatterns: [String] = [
        "[BLANK_AUDIO]", "(BLANK_AUDIO)",
        "[music]", "(music)", "[Music]", "(Music)",
        "[applause]", "(applause)",
        "[laughter]", "(laughter)",
        "ご視聴ありがとうございました",
        "お疲れ様でした",
        "おやすみなさい",
        "Thank you for watching",
        "Thanks for watching",
        "Subtitles by",
        "MoistCr1TiKaL",
    ]

    private func isHallucination(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)

        // Check known hallucination phrases
        for pattern in Self.hallucinationPatterns {
            if trimmed.localizedCaseInsensitiveContains(pattern) {
                return true
            }
        }

        // Filter bracketed stage directions only (avoid dropping real speech in brackets).
        if (trimmed.hasPrefix("[") && trimmed.hasSuffix("]")) ||
           (trimmed.hasPrefix("(") && trimmed.hasSuffix(")")) {
            let inner = String(trimmed.dropFirst().dropLast()).trimmingCharacters(in: .whitespacesAndNewlines)
            let stageDirectionKeywords = [
                "音", "無音", "ノイズ", "music", "applause", "laughter", "blank_audio"
            ]
            if stageDirectionKeywords.contains(where: { inner.localizedCaseInsensitiveContains($0) }) {
                return true
            }
        }

        // Filter highly repetitive text (same short phrase repeated 3+ times)
        if trimmed.count > 6 {
            let chars = Array(trimmed)
            for patternLen in 1...min(chars.count / 3, 20) {
                let sub = String(chars[0..<patternLen])
                let repetitions = trimmed.components(separatedBy: sub).count - 1
                if repetitions >= 3 && sub.count * repetitions >= trimmed.count / 2 {
                    return true
                }
            }
        }

        return false
    }

    private func handleTranscriptionUpdate(text: String) {
        let cleaned = text.trimmingCharacters(in: .whitespacesAndNewlines)
        print("📝 Live Transcript: \"\(cleaned)\"")

        guard !cleaned.isEmpty else { return }

        // Filter out whisper hallucinations
        if isHallucination(cleaned) {
            print("🚫 Filtered hallucination: \(cleaned.prefix(60))")
            return
        }

        let mergedText = mergeStreamingText(existing: currentBubbleText, incoming: cleaned)
        currentBubbleText = mergedText
        currentText = mergedText

        upsertCurrentUserBubble(with: mergedText)

        silenceTimer?.invalidate()
        silenceTimer = Timer.scheduledTimer(withTimeInterval: silenceThreshold, repeats: false) { [weak self] _ in
            Task { @MainActor in
                self?.finalizeCurrentBubble()
            }
        }
    }

    private func upsertCurrentUserBubble(with text: String) {
        if let lastIndex = conversation.indices.last,
           conversation[lastIndex].isUser,
           !currentBubbleFinalized {
            let existing = conversation[lastIndex]
            let updatedItem = ConversationItem(
                id: existing.id,
                text: text,
                isUser: existing.isUser,
                timestamp: existing.timestamp
            )
            var updatedConversation = conversation
            updatedConversation[lastIndex] = updatedItem
            conversation = updatedConversation
            return
        }

        startNewUserBubble(with: text)
        currentBubbleFinalized = false
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

    private func startNewUserBubble(with text: String) {
        let item = ConversationItem(
            id: UUID(),
            text: text,
            isUser: true,
            timestamp: Date()
        )
        conversation = conversation + [item]
    }

    private func finalizeCurrentBubble() {
        silenceTimer?.invalidate()
        silenceTimer = nil
        currentBubbleFinalized = true
        currentText = ""
        currentBubbleText = ""
    }

    private func requestMicrophonePermission() async -> Bool {
        if ProcessInfo.processInfo.isiOSAppOnMac {
            // On macOS (running as iPad app), use AVCaptureDevice for permission
            let status = AVCaptureDevice.authorizationStatus(for: .audio)
            switch status {
            case .authorized:
                print("🎤 Mic permission: already authorized (Mac)")
                return true
            case .notDetermined:
                let granted = await AVCaptureDevice.requestAccess(for: .audio)
                print("🎤 Mic permission: user responded \(granted) (Mac)")
                return granted
            case .denied, .restricted:
                print("🎤 Mic permission: denied/restricted (Mac)")
                return false
            @unknown default:
                return false
            }
        } else {
            // On iOS, use AVAudioSession
            return await withCheckedContinuation { continuation in
                AVAudioSession.sharedInstance().requestRecordPermission { granted in
                    print("🎤 Mic permission: \(granted) (iOS)")
                    continuation.resume(returning: granted)
                }
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

        // For streaming, the incoming text from the sliding window typically
        // represents the latest understanding of what was said.
        // Prefer the incoming result as it has the most recent context.
        if incoming.hasPrefix(existing) {
            return incoming
        }

        if incoming.contains(existing) {
            return incoming
        }

        if existing.contains(incoming) {
            return existing
        }

        // For Japanese: try character-level overlap (no spaces between words)
        let overlap = bestOverlapLength(suffixSource: existing, prefixSource: incoming)
        if overlap > 0 {
            let appendStart = incoming.index(incoming.startIndex, offsetBy: overlap)
            return existing + String(incoming[appendStart...])
        }

        // No overlap found — append directly (no space for Japanese)
        let isJapanese = selectedLanguage == "ja" || selectedLanguage == "zh" || selectedLanguage == "ko"
        let separator = isJapanese ? "" : " "
        return (existing + separator + incoming).trimmingCharacters(in: .whitespacesAndNewlines)
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
                    VStack(spacing: 8) {
                        // Show live partial transcript while streaming
                        if !streamer.currentText.isEmpty {
                            HStack {
                                Spacer()
                                Text(streamer.currentText)
                                    .font(.body)
                                    .foregroundColor(.white.opacity(0.9))
                                    .padding(12)
                                    .background(Color.green.opacity(0.5))
                                    .cornerRadius(16)
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 16)
                                            .stroke(Color.green.opacity(0.3), lineWidth: 1)
                                    )
                            }
                            .padding(.horizontal)
                        }

                        HStack(spacing: 6) {
                            Circle()
                                .fill(Color.green)
                                .frame(width: 10, height: 10)
                                .opacity(0.8)
                            Text("Listening...")
                                .font(.caption)
                                .foregroundColor(.green.opacity(0.8))
                        }
                        .padding(.vertical, 8)
                    }
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
