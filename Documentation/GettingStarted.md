# Getting Started with DTLN-aec CoreML

This guide will help you integrate neural echo cancellation into your iOS or macOS app.

## Installation

### Swift Package Manager

Add DTLNAecCoreML to your project using Xcode:

1. Open your project in Xcode
2. Go to **File → Add Package Dependencies**
3. Enter the repository URL: `https://github.com/anthropics/dtln-aec-coreml.git`
4. Select the version and click **Add Package**

Or add it to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/anthropics/dtln-aec-coreml.git", from: "0.4.0-beta")
]
```

## Quick Start

### Basic Usage

```swift
import DTLNAecCoreML
import DTLNAec256  // Import the model package you need

// 1. Create the processor (256-unit model recommended for most apps)
let processor = DTLNAecEchoProcessor(modelSize: .medium)

// 2. Load the models from the model package bundle
try processor.loadModels(from: DTLNAec256.bundle)

// 3. In your audio processing callback:
// Feed the far-end (speaker) audio as reference
processor.feedFarEnd(speakerSamples)

// Process the near-end (microphone) audio
let cleanAudio = processor.processNearEnd(microphoneSamples)

// 4. When recording ends, flush remaining buffered audio
let remaining = processor.flush()
processor.resetStates()  // Reset for next session
```

### Using Configuration

For more control, use `DTLNAecConfig`:

```swift
import DTLNAec512

var config = DTLNAecConfig()
config.modelSize = .large  // Best quality
config.computeUnits = .cpuAndNeuralEngine  // Use Neural Engine when available
config.enablePerformanceTracking = true

let processor = DTLNAecEchoProcessor(config: config)
try processor.loadModels(from: DTLNAec512.bundle)
```

### Async Model Loading

Load models without blocking the main thread:

```swift
import DTLNAec256

Task {
    let processor = DTLNAecEchoProcessor(modelSize: .medium)
    try await processor.loadModelsAsync(from: DTLNAec256.bundle)
    // Models are ready
}
```

## Choosing a Model Size

All models have **~32ms end-to-end latency** (fixed, from STFT buffering).

| Model | Quality | Processing Overhead | Use Case |
|-------|---------|---------------------|----------|
| `.small` (128 units) | Good | <1ms | Smallest bundle size |
| `.medium` (256 units) | Great | <2ms | **Recommended for most apps** |
| `.large` (512 units) | Best | <3ms | Best quality for long audio |

All models run well within the 8ms real-time budget. The 256-unit model is recommended for most applications due to its fast convergence (~0.3s) and 50dB echo suppression.

## What's Next?

- [Audio Requirements](AudioRequirements.md) - Sample rate, format, and buffering
- [API Reference](API.md) - Complete API documentation
- [Benchmarking](Benchmarking.md) - Measure performance on your device
