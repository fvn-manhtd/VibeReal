
# VibeReal

VibeReal is a Swift-based application integrating WhisperKit for speech-to-text capabilities.

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
   Xcode should automatically begin resolving Swift Package Manager dependencies (including `WhisperKit`). Wait for this process to finish.

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

- **Model Download**: The application is currently configured to use the `large-v3` model. On the first run, this model will be downloaded, which may take some time depending on your internet connection.
- **Performance**: The `large-v3` model is resource-intensive. Running on a real device with a Neural Engine (Apple Silicon) is recommended for better performance compared to the simulator.
