# Audio Requirements

DTLN-aec has specific audio format requirements. This guide covers what you need to know.

## Required Format

| Property | Value |
|----------|-------|
| Sample Rate | **16,000 Hz** (16kHz) |
| Channels | **Mono** |
| Format | **Float32** (`[Float]` in Swift) |

## Timing

| Property | Value | Meaning |
|----------|-------|---------|
| Block Size | 512 samples | 32ms internal processing window |
| Block Shift | 128 samples | 8ms frame size |
| End-to-End Latency | ~32ms | Algorithmic delay from STFT buffering |
| Processing Overhead | <2ms | Neural network inference time |

## Sample Rate Conversion

If your audio is at a different sample rate (e.g., 48kHz), you must resample to 16kHz before processing:

```swift
import AVFoundation
import Accelerate

// Example: Convert 48kHz to 16kHz
func resample48kTo16k(_ input: [Float]) -> [Float] {
    let inputRate: Double = 48000
    let outputRate: Double = 16000
    let ratio = inputRate / outputRate  // 3.0

    let outputLength = Int(Double(input.count) / ratio)
    var output = [Float](repeating: 0, count: outputLength)

    // Simple decimation (for production, use proper anti-aliasing)
    vDSP_vgenp(input, vDSP_Stride(1),
               [Float](stride(from: 0, to: Float(input.count), by: Float(ratio))),
               vDSP_Stride(1),
               &output, vDSP_Stride(1),
               vDSP_Length(outputLength),
               vDSP_Length(input.count))

    return output
}
```

For production use, consider using `AVAudioConverter` or a proper resampling library with anti-aliasing filters.

## Buffering Strategy

The processor accumulates samples internally and processes them in 512-sample blocks with 128-sample shifts. You can feed samples in any chunk size:

```swift
// Feed samples in various chunk sizes - all work correctly
processor.feedFarEnd(samples)  // Any size works

// Process and get output
let output = processor.processNearEnd(micSamples)
// Output may be smaller than input due to buffering
```

### Understanding Latency

**End-to-end latency is ~32ms**, determined by the STFT block size (512 samples at 16kHz). This is the same for all model sizes.

- **Algorithmic latency**: ~32ms (from STFT buffering)
- **Processing overhead**: <2ms on Apple Silicon (well within the 8ms real-time budget)
- **Audio pipeline latency**: Additional delay from your app's audio callbacks

### End of Recording

When a recording ends, call `flush()` to retrieve any remaining buffered audio (up to 511 samples / ~32ms):

```swift
// Get remaining buffered audio at end of recording
let remaining = processor.flush()

// Reset for next session
processor.resetStates()
```

## Far-End vs Near-End

- **Far-end (loopback)**: Audio playing through the speaker that might echo back
- **Near-end (microphone)**: Audio captured from the user

For best results:
1. Feed far-end audio **before** processing near-end
2. Ensure both streams are synchronized in time
3. The far-end buffer automatically manages up to ~500ms of history

## Stereo to Mono

If your audio is stereo, convert to mono before processing:

```swift
func stereoToMono(_ left: [Float], _ right: [Float]) -> [Float] {
    var mono = [Float](repeating: 0, count: left.count)
    vDSP_vadd(left, 1, right, 1, &mono, 1, vDSP_Length(left.count))
    vDSP_vsmul(mono, 1, [Float(0.5)], &mono, 1, vDSP_Length(left.count))
    return mono
}
```

## Related

- [Getting Started](GettingStarted.md)
- [API Reference](API.md)
