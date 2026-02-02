import XCTest

@testable import DTLNAecCoreML

/// Tests that verify actual echo cancellation quality using synthetic audio
final class AECQualityTests: XCTestCase {

  static let sampleRate: Float = 16000

  // MARK: - Test Scenarios

  /// Pure echo scenario: mic contains ONLY the echo of far-end signal
  /// Expected: output should be significantly attenuated
  func testPureEchoScenario() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()

    let duration: Float = 2.0  // 2 seconds
    let numSamples = Int(Self.sampleRate * duration)

    // Generate far-end signal: 440Hz + 880Hz sine wave (simulates music/voice)
    var farEnd = [Float](repeating: 0, count: numSamples)
    for i in 0..<numSamples {
      let t = Float(i) / Self.sampleRate
      farEnd[i] = 0.3 * sin(2 * .pi * 440 * t) + 0.2 * sin(2 * .pi * 880 * t)
    }

    // Simulate echo: delayed and attenuated version of far-end
    // Typical room echo: 20-100ms delay, 0.3-0.6 attenuation
    let echoDelayMs: Float = 50
    let echoAttenuation: Float = 0.4
    let echoDelaySamples = Int(echoDelayMs * Self.sampleRate / 1000)

    var nearEnd = [Float](repeating: 0, count: numSamples)
    for i in echoDelaySamples..<numSamples {
      nearEnd[i] = farEnd[i - echoDelaySamples] * echoAttenuation
    }

    // Process
    processor.feedFarEnd(farEnd)
    let output = processor.processNearEnd(nearEnd)

    // Measure echo reduction
    let inputEnergy = computeEnergy(nearEnd)
    let outputEnergy = computeEnergy(output)
    let reductionDb = 10 * log10(inputEnergy / max(outputEnergy, 1e-10))

    print("Pure Echo Test:")
    print("  Input energy: \(inputEnergy)")
    print("  Output energy: \(outputEnergy)")
    print("  Echo reduction: \(String(format: "%.1f", reductionDb)) dB")

    // Expect at least 10dB reduction for pure echo
    XCTAssertGreaterThan(reductionDb, 10, "Expected >10dB echo reduction")
  }

  /// Echo with near-end talker: mic contains both echo AND desired speech
  /// This test measures whether the model preserves energy during speech regions
  /// Uses chunked processing for realistic behavior
  func testEchoWithNearEndTalker() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()

    let duration: Float = 3.0
    let numSamples = Int(Self.sampleRate * duration)

    // Far-end: 440Hz tone (represents system audio)
    var farEnd = [Float](repeating: 0, count: numSamples)
    for i in 0..<numSamples {
      let t = Float(i) / Self.sampleRate
      farEnd[i] = 0.4 * sin(2 * .pi * 440 * t)
    }

    // Near-end talker: different frequency (300Hz) - represents user's voice
    // Active only in middle portion (1.0s - 2.0s) to create distinct regions
    var nearEndSpeech = [Float](repeating: 0, count: numSamples)
    for i in 0..<numSamples {
      let t = Float(i) / Self.sampleRate
      if t >= 1.0 && t <= 2.0 {
        nearEndSpeech[i] = 0.5 * sin(2 * .pi * 300 * t)
      }
    }

    // Echo of far-end (constant throughout)
    let echoDelaySamples = Int(40 * Self.sampleRate / 1000)
    var echo = [Float](repeating: 0, count: numSamples)
    for i in echoDelaySamples..<numSamples {
      echo[i] = farEnd[i - echoDelaySamples] * 0.35
    }

    // Microphone = echo + near-end speech
    var nearEnd = [Float](repeating: 0, count: numSamples)
    for i in 0..<numSamples {
      nearEnd[i] = echo[i] + nearEndSpeech[i]
    }

    // Process in chunks (simulating real-time processing)
    let chunkSize = 128
    var output: [Float] = []

    for start in stride(from: 0, to: numSamples, by: chunkSize) {
      let end = min(start + chunkSize, numSamples)
      let farEndChunk = Array(farEnd[start..<end])
      let nearEndChunk = Array(nearEnd[start..<end])

      processor.feedFarEnd(farEndChunk)
      let processed = processor.processNearEnd(nearEndChunk)
      output.append(contentsOf: processed)
    }

    print("Echo + Near-End Talker Test:")
    print("  Input samples: \(numSamples), Output samples: \(output.count)")

    // Analyze energy in different time regions
    // Echo-only region: 0.3s-0.8s (before speech)
    let echoOnlyStart = Int(0.3 * Self.sampleRate)
    let echoOnlyEnd = Int(0.8 * Self.sampleRate)
    let echoOnlyEnergy = computeEnergy(
      Array(output[min(echoOnlyStart, output.count)..<min(echoOnlyEnd, output.count)]))

    // Speech+echo region: 1.2s-1.8s (during speech)
    let speechStart = Int(1.2 * Self.sampleRate)
    let speechEnd = Int(1.8 * Self.sampleRate)
    let speechRegionEnergy = computeEnergy(
      Array(output[min(speechStart, output.count)..<min(speechEnd, output.count)]))

    // Echo-only region after speech: 2.2s-2.7s
    let echoOnly2Start = Int(2.2 * Self.sampleRate)
    let echoOnly2End = min(Int(2.7 * Self.sampleRate), output.count)
    let echoOnly2Energy = computeEnergy(
      Array(output[min(echoOnly2Start, output.count)..<echoOnly2End]))

    print("  Echo-only region (before speech) energy: \(String(format: "%.6f", echoOnlyEnergy))")
    print("  Speech+echo region energy: \(String(format: "%.6f", speechRegionEnergy))")
    print("  Echo-only region (after speech) energy: \(String(format: "%.6f", echoOnly2Energy))")

    let avgEchoOnlyEnergy = (echoOnlyEnergy + echoOnly2Energy) / 2
    print(
      "  Speech/echo energy ratio: \(String(format: "%.2f", speechRegionEnergy / max(avgEchoOnlyEnergy, 1e-10)))"
    )

    // Speech region should have more energy than echo-only regions
    XCTAssertGreaterThan(
      speechRegionEnergy, avgEchoOnlyEnergy,
      "Speech region should have more energy than echo-only regions")
  }

  /// Test with varying echo delays
  func testVaryingEchoDelays() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()

    let delays: [Float] = [10, 30, 50, 100, 150]  // milliseconds
    let duration: Float = 1.0
    let numSamples = Int(Self.sampleRate * duration)

    print("\nVarying Echo Delay Test:")
    print("  Delay (ms) | Reduction (dB)")
    print("  -----------+-----------------")

    for delayMs in delays {
      processor.resetStates()

      // Far-end signal
      var farEnd = [Float](repeating: 0, count: numSamples)
      for i in 0..<numSamples {
        let t = Float(i) / Self.sampleRate
        farEnd[i] = 0.4 * sin(2 * .pi * 500 * t)
      }

      // Echo with this delay
      let delaySamples = Int(delayMs * Self.sampleRate / 1000)
      var nearEnd = [Float](repeating: 0, count: numSamples)
      for i in delaySamples..<numSamples {
        nearEnd[i] = farEnd[i - delaySamples] * 0.4
      }

      processor.feedFarEnd(farEnd)
      let output = processor.processNearEnd(nearEnd)

      let inputEnergy = computeEnergy(nearEnd)
      let outputEnergy = computeEnergy(output)
      let reductionDb = 10 * log10(inputEnergy / max(outputEnergy, 1e-10))

      print("  \(String(format: "%7.0f", delayMs)) ms | \(String(format: "%8.1f", reductionDb)) dB")
    }
  }

  /// Test with broadband noise as far-end (more realistic)
  func testBroadbandEcho() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()

    let duration: Float = 2.0
    let numSamples = Int(Self.sampleRate * duration)

    // Far-end: sum of multiple frequencies (approximates broadband)
    var farEnd = [Float](repeating: 0, count: numSamples)
    let frequencies: [Float] = [200, 400, 600, 800, 1000, 1500, 2000, 3000]
    for i in 0..<numSamples {
      let t = Float(i) / Self.sampleRate
      for freq in frequencies {
        farEnd[i] += 0.05 * sin(2 * .pi * freq * t + Float.random(in: 0...(.pi * 2)))
      }
    }

    // Echo
    let echoDelaySamples = Int(60 * Self.sampleRate / 1000)
    var nearEnd = [Float](repeating: 0, count: numSamples)
    for i in echoDelaySamples..<numSamples {
      nearEnd[i] = farEnd[i - echoDelaySamples] * 0.3
    }

    // Process in streaming fashion to avoid ring buffer overflow
    let chunkSize = 512
    var output: [Float] = []
    for start in stride(from: 0, to: numSamples, by: chunkSize) {
      let end = min(start + chunkSize, numSamples)
      processor.feedFarEnd(Array(farEnd[start..<end]))
      let processed = processor.processNearEnd(Array(nearEnd[start..<end]))
      output.append(contentsOf: processed)
    }

    // Skip the echo delay period + warmup to measure steady-state suppression
    // Echo starts at echoDelaySamples, add 2000 samples (~125ms) for model warmup
    let measureStart = echoDelaySamples + 2000
    let inputSteadyState = Array(nearEnd[measureStart...])
    let outputSteadyState = output.count > measureStart ? Array(output[measureStart...]) : output

    let inputEnergy = computeEnergy(inputSteadyState)
    let outputEnergy = computeEnergy(outputSteadyState)
    let reductionDb = 10 * log10(inputEnergy / max(outputEnergy, 1e-10))

    print("\nBroadband Echo Test:")
    print("  Echo reduction: \(String(format: "%.1f", reductionDb)) dB (steady-state)")
    print("  Input samples: \(nearEnd.count), Output samples: \(output.count)")
    print("  Measuring from sample \(measureStart)")

    // Verify output quality
    // 1. Should produce reasonable output count
    XCTAssertGreaterThan(
      output.count, numSamples - 256,
      "Output should have approximately the same number of samples as input")

    // 2. Output should not contain NaN or Inf
    let hasInvalidValues = output.contains { $0.isNaN || $0.isInfinite }
    XCTAssertFalse(hasInvalidValues, "Output should not contain NaN or Inf values")

    // 3. Echo reduction threshold
    // Note: Synthetic multi-frequency signals with random phases may not perform as well as real audio.
    // Real audio tests (see testCompareWithPythonReference) show excellent suppression
    // matching Python TFLite reference (49-53 dB).
    // Using 1dB as minimum for synthetic broadband signals (original threshold).
    XCTAssertGreaterThan(
      reductionDb, 1, "Expected >1dB reduction for broadband echo (steady-state)")
  }

  /// Double-talk scenario: simultaneous far-end and near-end
  func testDoubleTalk() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()

    let duration: Float = 2.0
    let numSamples = Int(Self.sampleRate * duration)

    // Far-end at 500Hz
    var farEnd = [Float](repeating: 0, count: numSamples)
    for i in 0..<numSamples {
      let t = Float(i) / Self.sampleRate
      farEnd[i] = 0.4 * sin(2 * .pi * 500 * t)
    }

    // Near-end speech at 250Hz (lower, different from echo)
    var nearEndSpeech = [Float](repeating: 0, count: numSamples)
    for i in 0..<numSamples {
      let t = Float(i) / Self.sampleRate
      nearEndSpeech[i] = 0.4 * sin(2 * .pi * 250 * t)
    }

    // Echo
    let echoDelaySamples = Int(50 * Self.sampleRate / 1000)
    var echo = [Float](repeating: 0, count: numSamples)
    for i in echoDelaySamples..<numSamples {
      echo[i] = farEnd[i - echoDelaySamples] * 0.35
    }

    // Mic = echo + speech (full duration double-talk)
    var nearEnd = [Float](repeating: 0, count: numSamples)
    for i in 0..<numSamples {
      nearEnd[i] = echo[i] + nearEndSpeech[i]
    }

    processor.feedFarEnd(farEnd)
    let output = processor.processNearEnd(nearEnd)

    // Check that output still has content (speech preserved)
    let outputEnergy = computeEnergy(output)
    let speechEnergy = computeEnergy(nearEndSpeech)

    print("\nDouble-Talk Test:")
    print("  Near-end speech energy: \(speechEnergy)")
    print("  Output energy: \(outputEnergy)")
    print("  Energy ratio: \(String(format: "%.2f", outputEnergy / speechEnergy))")

    // Output should retain meaningful energy (speech not suppressed)
    XCTAssertGreaterThan(
      outputEnergy, speechEnergy * 0.1, "Near-end speech should be partially preserved")
  }

  // MARK: - Helper Functions

  func computeEnergy(_ samples: [Float]) -> Float {
    guard !samples.isEmpty else { return 0 }
    return samples.reduce(0) { $0 + $1 * $1 } / Float(samples.count)
  }

  func computeCorrelation(_ a: [Float], _ b: [Float]) -> Float {
    let length = min(a.count, b.count)
    guard length > 0 else { return 0 }

    let aMean = a.prefix(length).reduce(0, +) / Float(length)
    let bMean = b.prefix(length).reduce(0, +) / Float(length)

    var numerator: Float = 0
    var aDenom: Float = 0
    var bDenom: Float = 0

    for i in 0..<length {
      let aDiff = a[i] - aMean
      let bDiff = b[i] - bMean
      numerator += aDiff * bDiff
      aDenom += aDiff * aDiff
      bDenom += bDiff * bDiff
    }

    let denom = sqrt(aDenom * bDenom)
    return denom > 0 ? numerator / denom : 0
  }
}
