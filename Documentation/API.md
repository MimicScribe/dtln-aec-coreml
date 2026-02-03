# API Reference

Complete API documentation for DTLN-aec CoreML.

## DTLNAecEchoProcessor

The main class for echo cancellation processing.

### Initialization

```swift
// Using model size directly
let processor = DTLNAecEchoProcessor(modelSize: .small)

// Using configuration object
let config = DTLNAecConfig(
    modelSize: .large,
    computeUnits: .cpuAndNeuralEngine,
    enablePerformanceTracking: true
)
let processor = DTLNAecEchoProcessor(config: config)
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `config` | `DTLNAecConfig` | The configuration used by this processor |
| `modelSize` | `DTLNAecModelSize` | The model size being used |
| `numUnits` | `Int` | Number of LSTM units (128, 256, or 512) |
| `isInitialized` | `Bool` | Whether models are loaded and ready |
| `pendingSampleCount` | `Int` | Number of samples buffered but not yet output |
| `averageFrameTimeMs` | `Double` | Average processing time per frame |

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `sampleRate` | 16,000.0 | Required audio sample rate (Hz) |
| `blockLen` | 512 | FFT block size (samples) |
| `blockShift` | 128 | Frame shift (samples) |
| `fftBins` | 257 | Number of frequency bins |

### Methods

#### loadModels(from:)

```swift
func loadModels(from bundle: Bundle? = nil) throws
```

Loads CoreML models from the specified bundle. Call once at startup.

**Parameters:**
- `bundle`: The bundle containing model resources (e.g., `DTLNAec256.bundle`). If nil, searches module and main bundles.

**Throws:** `DTLNAecError.modelNotFound` if models aren't in the bundle.

#### loadModelsAsync(from:)

```swift
@available(macOS 10.15, iOS 13.0, *)
func loadModelsAsync(from bundle: Bundle? = nil) async throws
```

Asynchronously loads models without blocking the main thread.

**Parameters:**
- `bundle`: The bundle containing model resources. If nil, searches module and main bundles.

#### feedFarEnd(_:)

```swift
func feedFarEnd(_ samples: [Float])
```

Feeds far-end (speaker/loopback) audio samples. Call before `processNearEnd`.

**Parameters:**
- `samples`: Audio samples at 16kHz, Float32 format

#### processNearEnd(_:)

```swift
func processNearEnd(_ samples: [Float]) -> [Float]
```

Processes near-end (microphone) samples and returns echo-cancelled output.

**Parameters:**
- `samples`: Microphone audio at 16kHz, Float32 format

**Returns:** Echo-cancelled audio samples. May be fewer samples than input due to buffering.

#### resetStates()

```swift
func resetStates()
```

Resets LSTM states and clears buffers. Call when starting a new recording session.

#### flush()

```swift
func flush() -> [Float]
```

Flushes remaining buffered audio at end of recording.

When a recording ends, there may be samples that haven't been output yet:
- Pending samples (0-127) waiting for enough samples to form a frame
- Overlap-add tail (384 samples) from previous frame processing

This method processes any pending samples (zero-padded if needed) and returns all remaining buffered audio.

**Returns:** Remaining buffered audio samples (up to 511 samples, ~32ms at 16kHz).

**Note:** LSTM states are NOT reset by this method, preserving session continuity. Call `resetStates()` separately if starting a new recording session.

---

## DTLNAecConfig

Configuration options for the echo processor.

```swift
public struct DTLNAecConfig: Sendable {
    var modelSize: DTLNAecModelSize
    var computeUnits: MLComputeUnits
    var enablePerformanceTracking: Bool
    var validateNumerics: Bool
    var clipOutput: Bool
    var enableLinkedGainControl: Bool
}
```

### Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `modelSize` | `DTLNAecModelSize` | `.small` | Model variant to use (`.medium` recommended) |
| `computeUnits` | `MLComputeUnits` | `.cpuAndNeuralEngine` | CoreML compute units |
| `enablePerformanceTracking` | `Bool` | `true` | Track `averageFrameTimeMs` |
| `validateNumerics` | `Bool` | `true` | Check for NaN/Inf in model output |
| `clipOutput` | `Bool` | `true` | Clamp output to [-1, 1] range |
| `enableLinkedGainControl` | `Bool` | `true` | Apply linked gain control to both streams (see [Linked Gain Control](LinkedGainControl.md)) |

### Compute Units

| Value | Description |
|-------|-------------|
| `.cpuOnly` | CPU only, most compatible |
| `.cpuAndGPU` | CPU and GPU |
| `.cpuAndNeuralEngine` | CPU and Neural Engine (recommended) |
| `.all` | All available compute units |

---

## DTLNAecModelSize

Available model sizes.

```swift
public enum DTLNAecModelSize: Int, CaseIterable, Sendable {
    case small = 128   // 1.8M params, <1ms processing
    case medium = 256  // 3.9M params, <2ms processing
    case large = 512   // 10.4M params, <3ms processing
}
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `units` | `Int` | Number of LSTM units (raw value) |
| `estimatedSizeMB` | `Double` | Approximate model file size |

---

## DTLNAecError

Errors that can occur during processing.

```swift
public enum DTLNAecError: Error, LocalizedError {
    case modelNotFound(String)
    case initializationFailed(String)
    case inferenceFailed(String)
}
```

### Cases

| Case | Description |
|------|-------------|
| `modelNotFound(name)` | Model file not found in bundle |
| `initializationFailed(reason)` | Failed to initialize processor |
| `inferenceFailed(reason)` | Model inference failed |

---

## LinkedGainController

Applies identical gain to both mic and system audio to preserve the echo relationship. Used internally by `DTLNAecEchoProcessor` when `enableLinkedGainControl` is true.

See [Linked Gain Control](LinkedGainControl.md) for detailed documentation.

```swift
public final class LinkedGainController {
    // Soft-knee parameters
    let threshold: Float      // Default: 0.5
    let kneeWidth: Float      // Default: 0.3
    let ratio: Float          // Default: 8.0

    // Gain limits
    let minGain: Float        // Default: 0.1
    let maxGain: Float        // Default: 2.0

    init(
        threshold: Float = 0.5,
        kneeWidth: Float = 0.3,
        ratio: Float = 8.0,
        minGain: Float = 0.1,
        maxGain: Float = 2.0,
        attackMs: Float = 1.0,
        releaseMs: Float = 100.0,
        frameMs: Float = 8.0
    )

    func computeLinkedGain(micPeak: Float, sysPeak: Float) -> Float
    func reset()
    func captureState() -> (fastEnvelope: Float, slowEnvelope: Float, currentGain: Float)
    func restoreState(_ state: (fastEnvelope: Float, slowEnvelope: Float, currentGain: Float))
}
```

### Methods

#### computeLinkedGain(micPeak:sysPeak:)

Computes the gain to apply to both streams based on the maximum peak of both inputs.

**Parameters:**
- `micPeak`: Peak absolute value of mic audio frame
- `sysPeak`: Peak absolute value of system/loopback audio frame

**Returns:** Gain value to apply to both streams (clamped to [minGain, maxGain])

#### reset()

Resets envelope and gain state. Called automatically by `DTLNAecEchoProcessor.resetStates()`.

#### captureState() / restoreState(_:)

Captures and restores internal state. Used internally by `flush()` to preserve state across zero-padded frame processing.

---

## Thread Safety

`DTLNAecEchoProcessor` is **NOT thread-safe**. Call all methods from a single thread or serial dispatch queue.

Example using a serial queue:

```swift
let processingQueue = DispatchQueue(label: "com.app.echo-processing")

processingQueue.async {
    processor.feedFarEnd(speakerSamples)
    let clean = processor.processNearEnd(micSamples)
    // Use clean audio...
}
```

---

## Usage Example

```swift
import DTLNAecCoreML
import DTLNAec256  // Import the model package

class AudioProcessor {
    private let echoProcessor: DTLNAecEchoProcessor
    private let processingQueue = DispatchQueue(label: "echo-processing")

    init() {
        var config = DTLNAecConfig()
        config.modelSize = .medium  // Recommended for most apps
        config.enablePerformanceTracking = true

        echoProcessor = DTLNAecEchoProcessor(config: config)
    }

    func start() async throws {
        try await echoProcessor.loadModelsAsync(from: DTLNAec256.bundle)
    }

    func processAudio(mic: [Float], speaker: [Float]) -> [Float] {
        processingQueue.sync {
            echoProcessor.feedFarEnd(speaker)
            return echoProcessor.processNearEnd(mic)
        }
    }

    func finishRecording() -> [Float] {
        processingQueue.sync {
            // Get any remaining buffered audio
            let remaining = echoProcessor.flush()
            // Reset for next session
            echoProcessor.resetStates()
            return remaining
        }
    }
}
```
