# Audio Samples

Test audio files for evaluating DTLN-aec echo cancellation quality.

## Model Comparison

| Model | Convergence | Suppression | Recommended |
|-------|-------------|-------------|-------------|
| 128-unit | ~1.0s | 49 dB | Smallest bundle |
| 256-unit | ~0.3s | 50 dB | **Yes — best balance** |
| 512-unit | ~0.9s | 53 dB | Best quality |

The 256-unit model is recommended for most applications due to its fast convergence.

## Directory Structure

### `aec_challenge/`

Samples from the Microsoft AEC Challenge dataset.

| File | Description |
|------|-------------|
| `farend_singletalk_lpb.wav` | Far-end (loopback) reference signal |
| `farend_singletalk_mic.wav` | Microphone recording with echo |
| `farend_singletalk_processed_python_128.wav` | Processed by 128-unit TFLite model (reference) |
| `farend_singletalk_processed_python_128.txt` | Metadata documenting Python reference provenance |
| `farend_singletalk_processed_coreml_128.wav` | Processed by CoreML 128-unit model |
| `farend_singletalk_processed_coreml_256.wav` | Processed by CoreML 256-unit model (recommended) |
| `farend_singletalk_processed_coreml_512.wav` | Processed by CoreML 512-unit model |

### `realworld/`

Real-world recordings made by playing audio through speakers and recording with microphone.

| File | Description |
|------|-------------|
| `test_lpb.wav` | Reference signal played through speakers |
| `test_mic.wav` | Microphone recording (contains echo) |
| `test_processed_python_128.wav` | Processed by 128-unit TFLite model (reference) |
| `test_processed_python_128.txt` | Metadata documenting Python reference provenance |
| `test_processed_coreml_128.wav` | Processed by CoreML 128-unit model |
| `test_processed_coreml_256.wav` | Processed by CoreML 256-unit model (recommended) |
| `test_processed_coreml_512.wav` | Processed by CoreML 512-unit model |

## Usage

### Compare outputs

```bash
# Listen to original vs processed (AEC challenge sample)
afplay Samples/aec_challenge/farend_singletalk_mic.wav                   # Original with echo
afplay Samples/aec_challenge/farend_singletalk_processed_coreml_256.wav  # CoreML 256-unit (recommended)

# Compare all model sizes
afplay Samples/aec_challenge/farend_singletalk_processed_coreml_128.wav  # CoreML 128-unit
afplay Samples/aec_challenge/farend_singletalk_processed_coreml_256.wav  # CoreML 256-unit
afplay Samples/aec_challenge/farend_singletalk_processed_coreml_512.wav  # CoreML 512-unit

# Compare real-world recordings
afplay Samples/realworld/test_mic.wav                   # Original with echo
afplay Samples/realworld/test_processed_coreml_256.wav  # CoreML 256-unit (recommended)
```

### Process your own files

```bash
swift run FileProcessor \
  --mic your_mic.wav \
  --ref your_reference.wav \
  --output cleaned.wav \
  --model large
```

## Audio Format

All files are:
- Sample rate: 16 kHz
- Channels: Mono
- Format: PCM (Int16 or Float32)
