
# VibeReal

VibeReal is a Swift-based application integrating `whisper.cpp` for real-time speech-to-text.

## Prerequisites

- **macOS**: Latest version recommended.
- **Xcode**: Latest version (supports Swift and SwiftUI).

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd VibeReal
   ```

2. **Open the Project**:
   Double-click on `VibeReal.xcodeproj` to open it in Xcode.

3. **Resolve Dependencies**:
   Xcode should automatically begin resolving Swift Package Manager dependencies (the official `whisper.spm` wrapper for `whisper.cpp`). Wait for this process to finish.

4. **Build and Run**:
   - Select your target (e.g., your connected iPhone, iPad, or "My Mac" if supported).
   - Press `Cmd + R` or click the **Run** button (play icon) in the top-left corner.
   - Alternatively, you can use the command line:

## Build from Command Line

To build the `VibeReal` scheme, run the following command in the project directory:

```bash
xcodebuild -scheme VibeReal -destination 'platform=macOS,variant=Designed for iPad' build
```

Or if you are running on an Apple Silicon Mac as a "Designed for iPhone/iPad" app:

```bash
xcodebuild -scheme VibeReal -destination 'platform=macOS,arch=arm64' build
```

Alternatively, to run on the active Mac:

```bash
xcodebuild -scheme VibeReal -destination 'platform=macOS' -allowProvisioningUpdates build
```

## Important Notes

- **Model Download**: The application downloads the selected GGML model from Hugging Face on first use and stores it in app support storage.
- **Japanese Real-Time**: The default language is `ja` and the default model is `ggml-small.bin` for low-latency Japanese streaming.
- **Performance**: For the highest Japanese accuracy, use `ggml-medium.bin` or `ggml-large-v3-turbo.bin` on a physical Apple Silicon device.
