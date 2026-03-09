# DTLN-aec CoreML

[![CI](https://github.com/adamsro/dtln-aec-coreml/actions/workflows/ci.yml/badge.svg)](https://github.com/adamsro/dtln-aec-coreml/actions/workflows/ci.yml)

> **Note:** This is a community port, not affiliated with the original DTLN-aec authors. While regression tests verify near-comparable performance, subtle differences from the TensorFlow implementation may exist.

Neural acoustic echo cancellation for Apple platforms using CoreML.

This package provides a Swift wrapper for [DTLN-aec](https://github.com/breizhn/DTLN-aec), a dual-signal transformation LSTM network that placed **3rd in the Microsoft AEC Challenge 2021**.

**[Watch the demo video →](https://www.youtube.com/watch?v=p9-TGge1_EQ)**

## Features

- Real-time echo cancellation with ~32ms end-to-end latency
- 50dB echo suppression with the 256-unit model
- Sub-millisecond neural network inference on Apple Silicon
- Three model sizes: 128 units (~7 MB), 256 units (~15 MB), 512 units (~40 MB)
- **Separate model packages** — only bundle the model size you need
- Modern Swift API with async/await support
- Configurable compute units (CPU, GPU, Neural Engine)
- iOS 16+ and macOS 13+ support

## Installation

### Swift Package Manager

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/anthropics/dtln-aec-coreml.git", from: "0.4.0-beta")
],
targets: [
    .target(
        name: "YourApp",
        dependencies: [
            "DTLNAecCoreML",   // Core processing code
            "DTLNAec256",      // Medium model (~15 MB) - recommended for most apps
            // "DTLNAec128",   // Small model (~7 MB) - smaller bundle
            // "DTLNAec512",   // Large model (~40 MB) - best quality
        ]
    )
]
```

Or in Xcode: File → Add Package Dependencies → Enter the repository URL, then select which products to include.

## Quick Start

```swift
import DTLNAecCoreML
import DTLNAec256  // Import the model package you need

// Initialize processor
let processor = DTLNAecEchoProcessor(modelSize: .medium)

// Load CoreML models from the model package bundle
try processor.loadModels(from: DTLNAec256.bundle)

// During audio processing:
processor.feedFarEnd(systemAudioSamples)  // [Float] at 16kHz
let cleanAudio = processor.processNearEnd(microphoneSamples)

// When recording ends, get remaining buffered audio
let remaining = processor.flush()
processor.resetStates()
```

### Async Model Loading

```swift
import DTLNAecCoreML
import DTLNAec512

let processor = DTLNAecEchoProcessor(modelSize: .large)
try await processor.loadModelsAsync(from: DTLNAec512.bundle)  // Non-blocking
```

### Configuration Options

```swift
import DTLNAecCoreML
import DTLNAec512

var config = DTLNAecConfig()
config.modelSize = .large                    // Best quality
config.computeUnits = .cpuAndNeuralEngine   // Use Neural Engine
config.enablePerformanceTracking = true

let processor = DTLNAecEchoProcessor(config: config)
try processor.loadModels(from: DTLNAec512.bundle)
```

## Latency

**End-to-end latency: ~32ms** — This is the delay users experience, determined by the STFT buffering required for frequency-domain processing (512 samples at 16kHz). This is the same for all model sizes.

**Processing overhead: <2ms** — The 256-unit model processes each 8ms audio frame in under 2ms on Apple Silicon, well within real-time requirements.

### Model Comparison

| Model | Units | Bundle Size | Processing (P99) | Convergence | Suppression | Use Case |
|-------|-------|-------------|------------------|-------------|-------------|----------|
| `.small` | 128 | ~7 MB | <1ms | ~1.0s | 49 dB | Smallest bundle |
| `.medium` | 256 | ~15 MB | <2ms | **~0.3s** | **50 dB** | **Recommended** |
| `.large` | 512 | ~40 MB | <3ms | ~0.9s | 53 dB | Best quality |

The 256-unit model is recommended for most applications — it has the fastest convergence and excellent suppression with a moderate bundle size.

**Import the corresponding model package:**
- `DTLNAec128` for `.small`
- `DTLNAec256` for `.medium`
- `DTLNAec512` for `.large`

## Audio Requirements

- **Sample rate:** 16,000 Hz
- **Channels:** Mono
- **Format:** Float32

If your audio is at a different sample rate, resample before processing.

## Documentation

- [Getting Started](Documentation/GettingStarted.md) - Installation and basic usage
- [Audio Requirements](Documentation/AudioRequirements.md) - Sample rates, formats, buffering
- [API Reference](Documentation/API.md) - Complete API documentation
- [Benchmarking](Documentation/Benchmarking.md) - Measure performance
- [Model Conversion](Documentation/ModelConversion.md) - Convert custom models

## Benchmarking

Run the included benchmark to measure performance on your hardware:

```bash
swift run dtln-benchmark        # Basic benchmark
swift run dtln-benchmark -n 1000  # More iterations
swift run dtln-benchmark --json   # JSON output for CI
```

Sample output on Apple M1:

```
| Model | Params | Load    | Avg     | P99     | RT Ratio | Status |
|-------|--------|---------|---------|---------|----------|--------|
| 128   | 1.8M   |   421ms |  0.04ms |  0.74ms |   0.01x  | ✅     |
| 256   | 3.9M   |   430ms |  0.07ms |  1.25ms |   0.01x  | ✅     |
| 512   | 10.4M  |   512ms |  0.18ms |  2.97ms |   0.02x  | ✅     |
```

## Testing Echo Cancellation

### Unit Tests

Run the synthetic AEC quality tests:

```bash
swift test --filter AECQualityTests
```

### Real-World Test

Test with actual speaker-to-microphone echo on your Mac:

```bash
# 1. Record: plays audio through speakers while recording from mic
swift Scripts/record_aec_test.swift \
  Tests/DTLNAecCoreMLTests/Fixtures/farend.wav \
  /tmp/recorded_nearend.wav

# 2. Process: run echo cancellation
swift run FileProcessor \
  --mic /tmp/recorded_nearend.wav \
  --ref Tests/DTLNAecCoreMLTests/Fixtures/farend.wav \
  --output /tmp/cleaned.wav

# 3. Compare before/after
afplay /tmp/recorded_nearend.wav  # With echo
afplay /tmp/cleaned.wav           # Echo cancelled
```

### Process Your Own Files

```bash
swift run FileProcessor \
  --mic your_mic_recording.wav \
  --ref your_system_audio.wav \
  --output cleaned.wav \
  --model small  # or 'large'
```

Input files must be 16kHz mono WAV. Convert with ffmpeg if needed:

```bash
ffmpeg -i input.wav -ar 16000 -ac 1 -c:a pcm_f32le output.wav
```

## How It Works

DTLN-aec uses a two-part architecture:

1. **Part 1 (Frequency Domain):** Takes magnitude spectra of mic and loopback signals, generates a frequency mask using LSTM layers

2. **Part 2 (Time Domain):** Refines the output using learned time-domain representations with Conv1D encoders and LSTM layers

Both parts maintain LSTM state across frames to capture temporal context.

## Credits

- **DTLN-aec:** [Nils L. Westhausen](https://github.com/breizhn/DTLN-aec) - Original TensorFlow implementation
- **Microsoft AEC Challenge 2021:** Competition where DTLN-aec placed 3rd

## Used By

- [MimicScribe](https://mimicscribe.app/) — Meeting transcription app with instant talking points for macOS

## License

MIT License - see [LICENSE](LICENSE) file.

The original DTLN-aec model weights are provided under MIT License by Nils L. Westhausen. See [ThirdPartyLicenses/](ThirdPartyLicenses/) for details.
