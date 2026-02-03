# Linked Gain Control

Linked gain control is an input normalization feature that applies identical gain to both microphone and system audio streams before processing. This preserves the echo-to-reference ratio critical for stable DTLN convergence.

## Why It Exists

The DTLN neural network learns to cancel echo by comparing the microphone signal (which contains echo) with the reference signal (system/loopback audio). The relationship between these signals—specifically the ratio of echo energy to reference energy—is critical for the model's effectiveness.

### The Problem

When input audio levels vary significantly (loud speakers, close-talking users, varying microphone gains), several issues can occur:

1. **Clipping distortion**: Loud audio clips at ±1.0, destroying the echo relationship
2. **Model instability**: The DTLN model was trained on normalized audio; extreme levels can cause poor convergence
3. **Independent AGC breaks echo cancellation**: If you apply separate automatic gain control to mic and system audio, the echo-to-reference ratio changes unpredictably, degrading cancellation quality

### The Solution

Linked gain control applies the **same** gain value to both streams simultaneously. This:

- Keeps both signals within the model's optimal input range
- Preserves the echo-to-reference ratio exactly
- Uses soft-knee compression for transparent gain reduction on loud transients

## How It Works

The gain controller uses a soft-knee compressor design with dual-path envelope tracking:

```
Input Level    →    Envelope Follower    →    Soft-Knee Curve    →    Gain
(max of mic/sys)    (fast attack,            (gentle compression      (applied to
                     slow release)            above threshold)         both streams)
```

### Algorithm Overview

1. **Peak detection**: The maximum peak from both mic and system audio determines the input level
2. **Envelope following**: A dual-path follower provides fast attack (~1ms) and slow release (~100ms)
3. **Soft-knee compression**: Gradual gain reduction above the threshold prevents harsh limiting
4. **Gain smoothing**: Additional smoothing prevents audible gain pumping

### Default Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Threshold | 0.5 (~-6dB) | Level where compression begins |
| Knee width | 0.3 | Soft transition region |
| Ratio | 8:1 | Compression ratio above threshold |
| Min gain | 0.1 (-20dB) | Gain floor to prevent silence |
| Max gain | 2.0 (+6dB) | Gain ceiling |
| Attack | ~1ms | Time to reduce gain |
| Release | ~100ms | Time to restore gain |

## Configuration

### Enabling/Disabling

Linked gain control is **enabled by default**. To disable it:

```swift
var config = DTLNAecConfig()
config.enableLinkedGainControl = false  // Disable gain control
let processor = DTLNAecEchoProcessor(config: config)
```

Or in the initializer:

```swift
let config = DTLNAecConfig(
    modelSize: .medium,
    enableLinkedGainControl: false
)
```

### When to Disable

You may want to disable linked gain control if:

- Your application already implements linked/stereo gain control upstream
- You have strictly controlled input levels (e.g., fixed hardware gain)
- You need bit-exact processing without any gain modification

### When to Keep Enabled (Recommended)

Keep linked gain control enabled when:

- Input levels vary (most real-world scenarios)
- Using built-in microphones with automatic gain
- Processing voice calls where volume varies by speaker
- You want maximum echo cancellation quality without tuning input levels

## Technical Details

### State Preservation

The gain controller maintains internal state (envelope values, current gain) across frames for smooth operation. This state is:

- **Preserved across `flush()` calls**: Calling `flush()` to get remaining audio does not corrupt the gain state, maintaining consistent volume when processing resumes
- **Reset by `resetStates()`**: Starting a new recording session resets the gain controller along with LSTM states

### Passthrough Behavior

When the neural network produces invalid output (NaN/Inf) and falls back to passthrough mode, the gain-adjusted microphone audio is returned (not the raw input). This ensures consistent volume levels even during fallback.

### Thread Safety

The gain controller is protected by the same lock as the main processor. No additional synchronization is needed when using the standard `feedFarEnd`/`processNearEnd` API.

## Advanced: Custom Gain Controller

For advanced use cases requiring different compression characteristics, you can implement your own gain control upstream and disable the built-in controller:

```swift
// Disable built-in gain control
var config = DTLNAecConfig(enableLinkedGainControl: false)
let processor = DTLNAecEchoProcessor(config: config)

// Apply your own linked gain before feeding to processor
func processAudio(mic: [Float], speaker: [Float]) -> [Float] {
    let gain = computeYourLinkedGain(mic: mic, speaker: speaker)
    let normalizedMic = mic.map { $0 * gain }
    let normalizedSpeaker = speaker.map { $0 * gain }

    processor.feedFarEnd(normalizedSpeaker)
    return processor.processNearEnd(normalizedMic)
}
```

**Important**: If implementing custom gain control, ensure:
1. The same gain is applied to both streams
2. Gain changes are smoothed to avoid artifacts
3. Output stays within [-1, 1] range

## See Also

- [API Reference](API.md) - Full API documentation
- [Audio Requirements](AudioRequirements.md) - Input format requirements
- [Getting Started](GettingStarted.md) - Quick start guide
