# DTLN-aec CoreML Examples

This directory contains example applications demonstrating how to use DTLN-aec CoreML.

## Examples

### CommandLineDemo

A simple demonstration with synthetic audio that shows basic API usage.

```bash
# Add to Package.swift first (see below), then:
swift run CommandLineDemo
```

### FileProcessor

Process WAV files to remove echo.

```bash
# Add to Package.swift first (see below), then:
swift run FileProcessor --mic mic_with_echo.wav --ref speaker.wav --output cleaned.wav
```

## Adding Examples to Package.swift

To run these examples, add them as executable targets in `Package.swift`:

```swift
.executableTarget(
    name: "CommandLineDemo",
    dependencies: ["DTLNAecCoreML"],
    path: "Examples/CommandLineDemo"
),
.executableTarget(
    name: "FileProcessor",
    dependencies: ["DTLNAecCoreML"],
    path: "Examples/FileProcessor"
),
```

## iOS Integration Example

For iOS apps using AVAudioEngine, here's a typical integration pattern:

```swift
import AVFoundation
import DTLNAecCoreML
import DTLNAec256  // Import the model package

class AudioManager {
    private let engine = AVAudioEngine()
    private let processor = DTLNAecEchoProcessor(modelSize: .medium)
    private let processingQueue = DispatchQueue(label: "echo-processing")

    func start() async throws {
        // Load models from the model package bundle
        try await processor.loadModelsAsync(from: DTLNAec256.bundle)

        // Configure audio session
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, options: [.defaultToSpeaker])
        try session.setPreferredSampleRate(16000)
        try session.setActive(true)

        // Get audio format
        let inputFormat = engine.inputNode.inputFormat(forBus: 0)

        // Install tap for microphone input
        engine.inputNode.installTap(onBus: 0, bufferSize: 128, format: inputFormat) { [weak self] buffer, time in
            self?.processAudioBuffer(buffer)
        }

        try engine.start()
    }

    private func processAudioBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData else { return }

        let samples = Array(UnsafeBufferPointer(
            start: channelData[0],
            count: Int(buffer.frameLength)
        ))

        processingQueue.async { [weak self] in
            guard let self = self else { return }

            // In a real app, you'd get the speaker output here
            // For this example, we assume no far-end reference
            // self.processor.feedFarEnd(speakerSamples)

            let cleaned = self.processor.processNearEnd(samples)

            // Use cleaned audio (e.g., send to WebRTC, record, etc.)
            // ...
        }
    }

    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()

        // Flush any remaining buffered audio
        let remaining = processor.flush()
        // Handle remaining samples if needed...

        processor.resetStates()
    }
}
```

## Notes

- All examples require 16kHz mono audio
- The processor is NOT thread-safe - use a serial queue
- Call `flush()` when recording ends to get remaining buffered audio (~32ms)
- Call `resetStates()` when starting a new session
- For best results, feed far-end audio slightly before processing near-end
