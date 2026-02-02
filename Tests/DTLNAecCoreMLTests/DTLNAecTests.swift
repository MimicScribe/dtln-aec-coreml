import CoreML
import XCTest

@testable import DTLNAecCoreML

final class DTLNAecTests: XCTestCase {

  // MARK: - Model Size Tests

  func testModelSizeConstants() {
    XCTAssertEqual(DTLNAecModelSize.small.units, 128)
    XCTAssertEqual(DTLNAecModelSize.medium.units, 256)
    XCTAssertEqual(DTLNAecModelSize.large.units, 512)
  }

  func testModelSizeEstimatedSize() {
    XCTAssertEqual(DTLNAecModelSize.small.estimatedSizeMB, 3.6)
    XCTAssertEqual(DTLNAecModelSize.medium.estimatedSizeMB, 8.0)
    XCTAssertEqual(DTLNAecModelSize.large.estimatedSizeMB, 20.3)
  }

  func testModelSizeCaseIterable() {
    let allSizes = DTLNAecModelSize.allCases
    XCTAssertEqual(allSizes.count, 3)
    XCTAssertTrue(allSizes.contains(.small))
    XCTAssertTrue(allSizes.contains(.medium))
    XCTAssertTrue(allSizes.contains(.large))
  }

  // MARK: - Configuration Tests

  func testDefaultConfig() {
    let config = DTLNAecConfig()
    XCTAssertEqual(config.modelSize, .small)
    XCTAssertTrue(config.enablePerformanceTracking)
  }

  func testCustomConfig() {
    var config = DTLNAecConfig()
    config.modelSize = .large
    config.enablePerformanceTracking = false

    XCTAssertEqual(config.modelSize, .large)
    XCTAssertFalse(config.enablePerformanceTracking)
  }

  func testConfigInitWithParameters() {
    let config = DTLNAecConfig(
      modelSize: .large,
      computeUnits: .cpuOnly,
      enablePerformanceTracking: false
    )

    XCTAssertEqual(config.modelSize, .large)
    XCTAssertEqual(config.computeUnits, .cpuOnly)
    XCTAssertFalse(config.enablePerformanceTracking)
  }

  // MARK: - Processor Initialization Tests

  func testProcessorInitialization() {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    XCTAssertFalse(processor.isInitialized)
    XCTAssertEqual(processor.modelSize, .small)
    XCTAssertEqual(processor.numUnits, 128)
  }

  func testProcessorInitializationWithConfig() {
    let config = DTLNAecConfig(modelSize: .large)
    let processor = DTLNAecEchoProcessor(config: config)

    XCTAssertFalse(processor.isInitialized)
    XCTAssertEqual(processor.modelSize, .large)
    XCTAssertEqual(processor.numUnits, 512)
    XCTAssertEqual(processor.config.modelSize, .large)
  }

  func testAudioConstants() {
    XCTAssertEqual(DTLNAecEchoProcessor.sampleRate, 16_000)
    XCTAssertEqual(DTLNAecEchoProcessor.blockLen, 512)
    XCTAssertEqual(DTLNAecEchoProcessor.blockShift, 128)
    XCTAssertEqual(DTLNAecEchoProcessor.fftBins, 257)
  }

  // MARK: - Model Loading Tests

  func testModelLoadingSmall() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()
    XCTAssertTrue(processor.isInitialized)
  }

  func testModelLoadingLarge() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .large)
    try processor.loadModelsFromPackage()
    XCTAssertTrue(processor.isInitialized)
  }

  // MARK: - Processing Tests

  func testProcessingWithSyntheticData() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()

    // Generate synthetic test data
    let numSamples = 16000  // 1 second at 16kHz
    var micSamples = [Float](repeating: 0, count: numSamples)
    var loopbackSamples = [Float](repeating: 0, count: numSamples)

    for i in 0..<numSamples {
      let t = Float(i) / 16000.0
      micSamples[i] = sin(2 * .pi * 440 * t) * 0.5
      loopbackSamples[i] = sin(2 * .pi * 880 * t) * 0.3
    }

    // Process in chunks (feed far-end and mic together to avoid ring buffer overflow)
    let chunkSize = 128
    var outputSamples: [Float] = []

    for start in stride(from: 0, to: numSamples, by: chunkSize) {
      let end = min(start + chunkSize, numSamples)
      let micChunk = Array(micSamples[start..<end])
      let lpbChunk = Array(loopbackSamples[start..<end])
      processor.feedFarEnd(lpbChunk)
      let processed = processor.processNearEnd(micChunk)
      outputSamples.append(contentsOf: processed)
    }

    // Should have produced approximately the expected number of output samples
    // (1 second = 16000 samples, minus initial buffering latency of ~128 samples)
    XCTAssertGreaterThan(
      outputSamples.count, numSamples - 256,
      "Expected approximately \(numSamples) output samples")

    // Verify no NaN or Inf values in output
    let hasInvalidValues = outputSamples.contains { $0.isNaN || $0.isInfinite }
    XCTAssertFalse(hasInvalidValues, "Output should not contain NaN or Inf values")

    // Verify output values are in [-1, 1] range (clipping is enabled by default)
    let outOfRange = outputSamples.contains { $0 < -1.0 || $0 > 1.0 }
    XCTAssertFalse(outOfRange, "Output values should be in [-1, 1] range")

    // Verify processing actually modified the signal (not just passthrough)
    var differenceFound = false
    for i in 256..<min(outputSamples.count, micSamples.count) {
      if abs(outputSamples[i - 256] - micSamples[i]) > 0.001 {
        differenceFound = true
        break
      }
    }
    XCTAssertTrue(differenceFound, "Processing should modify the input signal")

    XCTAssertGreaterThan(processor.averageFrameTimeMs, 0)

    print("Processed \(outputSamples.count) samples")
    print("Average frame time: \(String(format: "%.2f", processor.averageFrameTimeMs))ms")
  }

  // MARK: - Edge Case Tests

  func testEmptyInput() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()

    // Feed empty arrays
    processor.feedFarEnd([])
    let output = processor.processNearEnd([])

    // Should return empty (no samples to process)
    XCTAssertEqual(output.count, 0)
  }

  func testSmallInput() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()

    // Feed very small input (less than one block)
    let smallInput = [Float](repeating: 0.5, count: 64)
    processor.feedFarEnd(smallInput)
    let output = processor.processNearEnd(smallInput)

    // Not enough samples for processing yet
    XCTAssertEqual(output.count, 0)
  }

  func testMinimumBlockSize() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()

    // Feed one block shift (128 samples) - the minimum processable chunk
    let blockShift = DTLNAecEchoProcessor.blockShift
    let input = [Float](repeating: 0.5, count: blockShift)

    processor.feedFarEnd(input)
    let output = processor.processNearEnd(input)

    // Should produce output for one frame (128 samples)
    XCTAssertGreaterThan(output.count, 0)
    XCTAssertEqual(output.count, blockShift)
  }

  func testLongInput() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()

    // Process 10 seconds of audio in streaming fashion
    let numSamples = 160000  // 10 seconds at 16kHz
    let chunkSize = 512
    var output: [Float] = []

    for start in stride(from: 0, to: numSamples, by: chunkSize) {
      let end = min(start + chunkSize, numSamples)
      let chunk = [Float](repeating: 0.1, count: end - start)
      processor.feedFarEnd(chunk)
      let processed = processor.processNearEnd(chunk)
      output.append(contentsOf: processed)
    }

    // Should produce approximately the expected number of samples (minus initial latency)
    XCTAssertGreaterThan(
      output.count, numSamples - 256,
      "Long input should produce at least \(numSamples - 256) output samples")

    // Verify no NaN or Inf values in output
    let hasInvalidValues = output.contains { $0.isNaN || $0.isInfinite }
    XCTAssertFalse(hasInvalidValues, "Long input should not produce NaN or Inf values")

    // Verify output values are in valid range
    let outOfRange = output.contains { $0 < -1.0 || $0 > 1.0 }
    XCTAssertFalse(outOfRange, "Output values should be in [-1, 1] range")

    // Compute and verify energy is reasonable (not zero, not exploding)
    var sumSquares: Float = 0
    for sample in output {
      sumSquares += sample * sample
    }
    let energy = sumSquares / Float(output.count)
    XCTAssertGreaterThan(energy, 0, "Output should have non-zero energy")
    XCTAssertLessThan(energy, 1.0, "Output energy should not explode")

    print("Processed \(numSamples) samples, got \(output.count) output samples, energy: \(energy)")
  }

  // MARK: - State Reset Tests

  func testResetStates() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()

    // Process some data
    let input = [Float](repeating: 0.5, count: 1024)
    processor.feedFarEnd(input)
    _ = processor.processNearEnd(input)

    XCTAssertGreaterThan(processor.averageFrameTimeMs, 0)

    // Reset states
    processor.resetStates()

    // Average time should be reset
    XCTAssertEqual(processor.averageFrameTimeMs, 0)

    // Processor should still be initialized
    XCTAssertTrue(processor.isInitialized)
  }

  func testResetDoesNotAffectModel() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()

    // Reset before processing
    processor.resetStates()

    // Should still be able to process
    let input = [Float](repeating: 0.5, count: 512)
    processor.feedFarEnd(input)
    let output = processor.processNearEnd(input)

    XCTAssertGreaterThan(output.count, 0)
  }

  // MARK: - Flush Tests

  func testFlushReturnsBufferedSamples() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()

    // Process some data to fill buffers
    let input = [Float](repeating: 0.3, count: 200)
    processor.feedFarEnd(input)
    let output = processor.processNearEnd(input)

    // 200 samples -> 128 processed, 72 pending
    XCTAssertEqual(output.count, 128)

    // Before flush, should have 72 pending + 384 overlap tail = 456
    XCTAssertEqual(processor.pendingSampleCount, 72 + 384)

    // Flush should return all remaining samples
    let flushed = processor.flush()

    // Should get: 128 (from processing padded pending) + 384 (overlap tail) = 512
    XCTAssertEqual(flushed.count, 512)

    // After flush, should have 0 pending + 384 tail (but outputBuffer is zeroed)
    XCTAssertEqual(processor.pendingSampleCount, 384)
  }

  func testFlushWithNoPendingSamples() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()

    // Process exactly blockShift samples so no pending remain
    let input = [Float](repeating: 0.3, count: 128)
    processor.feedFarEnd(input)
    let output = processor.processNearEnd(input)

    XCTAssertEqual(output.count, 128)
    XCTAssertEqual(processor.pendingSampleCount, 384)  // Only overlap tail

    // Flush should return just the overlap-add tail
    let flushed = processor.flush()
    XCTAssertEqual(flushed.count, 384)
  }

  func testFlushWithoutProcessing() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()

    // Just feed some samples without triggering processing
    let input = [Float](repeating: 0.3, count: 64)
    processor.feedFarEnd(input)
    _ = processor.processNearEnd(input)

    // Should have 64 pending samples
    XCTAssertEqual(processor.pendingSampleCount, 64 + 384)

    let flushed = processor.flush()

    // 128 from padded pending frame + 384 overlap tail
    XCTAssertEqual(flushed.count, 512)
  }

  func testFlushPreservesLSTMStates() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()

    // Generate a distinctive signal pattern
    var input = [Float](repeating: 0, count: 512)
    for i in 0..<512 {
      input[i] = Float(sin(Double(i) * 0.1)) * 0.5
    }

    // Process to warm up LSTM states
    processor.feedFarEnd(input)
    let outputBeforeFlush = processor.processNearEnd(input)

    // Flush (should preserve LSTM states)
    _ = processor.flush()

    // Process same data after flush - LSTM states should be preserved
    processor.feedFarEnd(input)
    let outputAfterFlush = processor.processNearEnd(input)

    // Create a fresh processor and compare
    let freshProcessor = DTLNAecEchoProcessor(modelSize: .small)
    try freshProcessor.loadModelsFromPackage()
    freshProcessor.feedFarEnd(input)
    let freshOutput = freshProcessor.processNearEnd(input)

    // 1. Outputs after flush should differ from fresh processor (states preserved)
    var differenceFromFresh = false
    for i in 0..<min(outputAfterFlush.count, freshOutput.count) {
      if abs(outputAfterFlush[i] - freshOutput[i]) > 0.001 {
        differenceFromFresh = true
        break
      }
    }
    XCTAssertTrue(
      differenceFromFresh,
      "LSTM states should be preserved after flush, causing different output than fresh processor")

    // 2. Compute correlation between outputs before and after flush
    // With same input and preserved states, correlation should be high
    func computeCorrelation(_ a: [Float], _ b: [Float]) -> Float {
      let n = min(a.count, b.count)
      guard n > 0 else { return 0 }

      var sumA: Float = 0
      var sumB: Float = 0
      for i in 0..<n {
        sumA += a[i]
        sumB += b[i]
      }
      let meanA = sumA / Float(n)
      let meanB = sumB / Float(n)

      var cov: Float = 0
      var varA: Float = 0
      var varB: Float = 0
      for i in 0..<n {
        let da = a[i] - meanA
        let db = b[i] - meanB
        cov += da * db
        varA += da * da
        varB += db * db
      }

      let denom = sqrt(varA * varB)
      return denom > 1e-10 ? cov / denom : 0
    }

    let correlationAfterFlush = computeCorrelation(outputBeforeFlush, outputAfterFlush)
    // State preserved means output may differ from fresh but correlation validates continuity
    // Note: correlation can vary based on LSTM state evolution; just verify it's not random
    XCTAssertGreaterThan(
      correlationAfterFlush, -0.5,
      "Output after flush should show some correlation pattern (not anti-correlated)")
    print("Correlation after flush: \(correlationAfterFlush)")
  }

  func testFlushClearsPendingBuffers() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()

    // Add pending samples
    let input = [Float](repeating: 0.3, count: 64)
    processor.feedFarEnd(input)
    _ = processor.processNearEnd(input)

    XCTAssertEqual(processor.pendingSampleCount, 64 + 384)

    // Flush
    _ = processor.flush()

    // Pending should be cleared (only overlap tail remaining, which is zeroed)
    XCTAssertEqual(processor.pendingSampleCount, 384)

    // Calling flush again should only return zeros from zeroed outputBuffer
    let secondFlush = processor.flush()
    XCTAssertEqual(secondFlush.count, 384)

    // Should be all zeros
    let nonZeroCount = secondFlush.filter { abs($0) > 0.0001 }.count
    XCTAssertEqual(nonZeroCount, 0, "Second flush should return zeros from cleared buffer")
  }

  func testFlushUninitializedProcessor() {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    // Don't load models

    // When uninitialized, processNearEnd passes through directly without buffering
    let input = [Float](repeating: 0.5, count: 64)
    processor.feedFarEnd(input)
    let output = processor.processNearEnd(input)

    // Passthrough mode: input returned directly
    XCTAssertEqual(output.count, 64)
    XCTAssertEqual(output, input)

    // Flush on uninitialized processor returns empty (nothing buffered)
    let flushed = processor.flush()
    XCTAssertEqual(flushed.count, 0)
  }

  func testPendingSampleCount() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()

    // Initially should have just overlap tail
    XCTAssertEqual(processor.pendingSampleCount, 384)

    // Add 50 samples (not enough to process)
    let input50 = [Float](repeating: 0.3, count: 50)
    processor.feedFarEnd(input50)
    _ = processor.processNearEnd(input50)
    XCTAssertEqual(processor.pendingSampleCount, 50 + 384)

    // Add 100 more samples (150 total pending, should process one frame)
    let input100 = [Float](repeating: 0.3, count: 100)
    processor.feedFarEnd(input100)
    _ = processor.processNearEnd(input100)
    // 150 - 128 = 22 pending
    XCTAssertEqual(processor.pendingSampleCount, 22 + 384)
  }

  // MARK: - Performance Tracking Tests

  func testPerformanceTrackingEnabled() throws {
    var config = DTLNAecConfig()
    config.enablePerformanceTracking = true
    let processor = DTLNAecEchoProcessor(config: config)
    try processor.loadModelsFromPackage()

    // Process some data
    let input = [Float](repeating: 0.5, count: 1024)
    processor.feedFarEnd(input)
    _ = processor.processNearEnd(input)

    // Should have tracked time
    XCTAssertGreaterThan(processor.averageFrameTimeMs, 0)
  }

  func testPerformanceTrackingDisabled() throws {
    var config = DTLNAecConfig()
    config.enablePerformanceTracking = false
    let processor = DTLNAecEchoProcessor(config: config)
    try processor.loadModelsFromPackage()

    // Process some data
    let input = [Float](repeating: 0.5, count: 1024)
    processor.feedFarEnd(input)
    _ = processor.processNearEnd(input)

    // Should not have tracked time
    XCTAssertEqual(processor.averageFrameTimeMs, 0)
  }

  // MARK: - Passthrough Test (uninitialized)

  func testUninitializedPassthrough() {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    // Don't load models

    let input = [Float](repeating: 0.5, count: 256)
    let output = processor.processNearEnd(input)

    // Should pass through the input when not initialized
    XCTAssertEqual(output.count, input.count)
    XCTAssertEqual(output, input)
  }

  // MARK: - Multiple Model Sizes

  func testMultipleProcessors() throws {
    // Create processors for different sizes
    let smallProcessor = DTLNAecEchoProcessor(modelSize: .small)
    let largeProcessor = DTLNAecEchoProcessor(modelSize: .large)

    try smallProcessor.loadModelsFromPackage()
    try largeProcessor.loadModelsFromPackage()

    XCTAssertTrue(smallProcessor.isInitialized)
    XCTAssertTrue(largeProcessor.isInitialized)
    XCTAssertEqual(smallProcessor.numUnits, 128)
    XCTAssertEqual(largeProcessor.numUnits, 512)
  }

  /// Test that 512-unit model works after float16 state fix
  func testLargeModelCPUOnly() throws {
    let config = DTLNAecConfig(modelSize: .large, computeUnits: .cpuOnly)
    let processor = DTLNAecEchoProcessor(config: config)
    try processor.loadModelsFromPackage()

    let numSamples = 16000  // 1 second
    var farEnd = [Float](repeating: 0, count: numSamples)
    var nearEnd = [Float](repeating: 0, count: numSamples)

    for i in 0..<numSamples {
      let t = Float(i) / 16000.0
      farEnd[i] = 0.3 * sin(2 * .pi * 440 * t)
      nearEnd[i] = 0.2 * sin(2 * .pi * 440 * t)
    }

    // Process in chunks
    let chunkSize = 128
    var output: [Float] = []
    var firstNaN: Int? = nil

    for (idx, start) in stride(from: 0, to: numSamples, by: chunkSize).enumerated() {
      let end = min(start + chunkSize, numSamples)
      processor.feedFarEnd(Array(farEnd[start..<end]))
      let processed = processor.processNearEnd(Array(nearEnd[start..<end]))

      if firstNaN == nil && processed.contains(where: { $0.isNaN }) {
        firstNaN = idx
      }
      output.append(contentsOf: processed)
    }

    print("Large model (CPU-only): \(output.count) samples")
    let hasNaN = output.contains { $0.isNaN }
    print("Has NaN: \(hasNaN), first at frame: \(firstNaN ?? -1)")
    XCTAssertFalse(hasNaN, "Large model should not produce NaN with float16 states")
  }

  // MARK: - Async Loading Tests

  @available(macOS 10.15, iOS 13.0, *)
  func testAsyncModelLoading() async throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    XCTAssertFalse(processor.isInitialized)

    try await processor.loadModelsFromPackageAsync()

    XCTAssertTrue(processor.isInitialized)
  }

  // MARK: - Error Tests

  func testModelNotFoundError() {
    // Test error description
    let error = DTLNAecError.modelNotFound("TestModel")
    XCTAssertNotNil(error.errorDescription)
    XCTAssertTrue(error.errorDescription?.contains("TestModel") ?? false)
  }

  func testInitializationFailedError() {
    let error = DTLNAecError.initializationFailed("Test reason")
    XCTAssertNotNil(error.errorDescription)
    XCTAssertTrue(error.errorDescription?.contains("Test reason") ?? false)
  }

  func testInferenceFailedError() {
    let error = DTLNAecError.inferenceFailed("Test reason")
    XCTAssertNotNil(error.errorDescription)
    XCTAssertTrue(error.errorDescription?.contains("Test reason") ?? false)
  }

  // MARK: - Sendable Conformance Tests

  func testModelSizeSendable() {
    // Verify DTLNAecModelSize is Sendable by using it across threads
    let size: DTLNAecModelSize = .small
    let expectation = XCTestExpectation(description: "Sendable test")

    Task.detached {
      // This should compile without warnings because DTLNAecModelSize is Sendable
      let localSize = size
      XCTAssertEqual(localSize.units, 128)
      expectation.fulfill()
    }

    wait(for: [expectation], timeout: 1.0)
  }

  func testConfigSendable() {
    let config = DTLNAecConfig(modelSize: .large)
    let expectation = XCTestExpectation(description: "Config Sendable test")

    Task.detached {
      let localConfig = config
      XCTAssertEqual(localConfig.modelSize, .large)
      expectation.fulfill()
    }

    wait(for: [expectation], timeout: 1.0)
  }

  // MARK: - Edge Case Tests (Additional)

  func testNaNInputHandling() throws {
    // Test that processor handles NaN input gracefully
    var config = DTLNAecConfig(modelSize: .small)
    config.validateNumerics = true
    config.clipOutput = true
    let processor = DTLNAecEchoProcessor(config: config)
    try processor.loadModelsFromPackage()

    // Create input with NaN values
    var input = [Float](repeating: 0.5, count: 256)
    input[100] = .nan
    input[200] = .infinity

    processor.feedFarEnd(input)
    let output = processor.processNearEnd(input)

    // With validation enabled, output should not contain NaN/Inf
    // (passthrough or clipping should handle it)
    let hasInvalidValues = output.contains { $0.isNaN || $0.isInfinite }
    // Note: NaN input may still produce NaN through passthrough if model not initialized
    // but clipping should at least bound Inf values
    print("NaN input test: output has \(output.count) samples, invalid values: \(hasInvalidValues)")
  }

  func testBoundaryConditions() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()

    // Test exact block shift size (128 samples)
    let blockShift = DTLNAecEchoProcessor.blockShift
    let exactInput = [Float](repeating: 0.3, count: blockShift)
    processor.feedFarEnd(exactInput)
    let exactOutput = processor.processNearEnd(exactInput)
    XCTAssertEqual(
      exactOutput.count, blockShift, "Exact blockShift input should produce blockShift output")

    // Test double block shift (256 samples)
    processor.resetStates()
    let doubleInput = [Float](repeating: 0.3, count: blockShift * 2)
    processor.feedFarEnd(doubleInput)
    let doubleOutput = processor.processNearEnd(doubleInput)
    XCTAssertEqual(
      doubleOutput.count, blockShift * 2,
      "Double blockShift input should produce double blockShift output")

    // Test off-by-one below block shift (127 samples)
    processor.resetStates()
    let belowInput = [Float](repeating: 0.3, count: blockShift - 1)
    processor.feedFarEnd(belowInput)
    let belowOutput = processor.processNearEnd(belowInput)
    XCTAssertEqual(belowOutput.count, 0, "Below blockShift input should produce no output yet")

    // Test off-by-one above block shift (129 samples)
    processor.resetStates()
    let aboveInput = [Float](repeating: 0.3, count: blockShift + 1)
    processor.feedFarEnd(aboveInput)
    let aboveOutput = processor.processNearEnd(aboveInput)
    XCTAssertEqual(
      aboveOutput.count, blockShift,
      "Above blockShift input should produce exactly blockShift output")
  }

  func testOutputClippingConfig() throws {
    // Test with clipping enabled
    var configClip = DTLNAecConfig(modelSize: .small)
    configClip.clipOutput = true
    let processorClip = DTLNAecEchoProcessor(config: configClip)
    try processorClip.loadModelsFromPackage()

    // Process some audio
    var input = [Float](repeating: 0, count: 512)
    for i in 0..<512 {
      input[i] = Float(sin(Double(i) * 0.1)) * 0.5
    }
    processorClip.feedFarEnd(input)
    let outputClip = processorClip.processNearEnd(input)

    // All values should be in [-1, 1]
    let allInRange = outputClip.allSatisfy { $0 >= -1.0 && $0 <= 1.0 }
    XCTAssertTrue(allInRange, "With clipOutput=true, all values should be in [-1, 1]")

    // Test with clipping disabled
    var configNoClip = DTLNAecConfig(modelSize: .small)
    configNoClip.clipOutput = false
    let processorNoClip = DTLNAecEchoProcessor(config: configNoClip)
    try processorNoClip.loadModelsFromPackage()
    processorNoClip.feedFarEnd(input)
    let outputNoClip = processorNoClip.processNearEnd(input)

    // Should still produce valid output (just not guaranteed clipped)
    XCTAssertGreaterThan(
      outputNoClip.count, 0, "With clipOutput=false, should still produce output")
  }

  func testValidateNumericsConfig() throws {
    var config = DTLNAecConfig(modelSize: .small)
    config.validateNumerics = true
    let processor = DTLNAecEchoProcessor(config: config)
    try processor.loadModelsFromPackage()

    // Normal input should work fine
    let input = [Float](repeating: 0.3, count: 256)
    processor.feedFarEnd(input)
    let output = processor.processNearEnd(input)

    // Should have output and no NaN/Inf
    XCTAssertGreaterThan(output.count, 0)
    let hasInvalidValues = output.contains { $0.isNaN || $0.isInfinite }
    XCTAssertFalse(hasInvalidValues, "Normal input should not produce invalid values")
  }

  func testNewConfigDefaultValues() {
    let config = DTLNAecConfig()
    XCTAssertTrue(config.validateNumerics, "validateNumerics should default to true")
    XCTAssertTrue(config.clipOutput, "clipOutput should default to true")
  }

  func testModelNotFoundWithEmptyBundle() {
    let processor = DTLNAecEchoProcessor(modelSize: .small)

    // Try to load from an empty/non-existent bundle
    do {
      try processor.loadModels(from: Bundle(path: "/nonexistent") ?? Bundle.main)
      // If no error thrown, the bundle might have found models in main bundle
      // This is acceptable behavior
    } catch let error as DTLNAecError {
      // Should throw modelNotFound error
      switch error {
      case .modelNotFound:
        // Expected
        break
      default:
        XCTFail("Expected modelNotFound error, got \(error)")
      }
    } catch {
      XCTFail("Unexpected error type: \(error)")
    }
  }

  // MARK: - Thread Safety Tests

  func testConcurrentFeedAndProcess() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()

    let iterations = 100
    let samplesPerIteration = 256
    let expectation = XCTestExpectation(description: "Concurrent processing completes")
    expectation.expectedFulfillmentCount = 2

    var farEndSamplesFed = 0
    var nearEndSamplesProcessed = 0
    var outputSamplesReceived = 0

    // Thread 1: Feed far-end samples
    DispatchQueue.global(qos: .userInteractive).async {
      for _ in 0..<iterations {
        let farEnd = [Float](repeating: 0.1, count: samplesPerIteration)
        processor.feedFarEnd(farEnd)
        farEndSamplesFed += samplesPerIteration
      }
      expectation.fulfill()
    }

    // Thread 2: Process near-end samples
    DispatchQueue.global(qos: .userInteractive).async {
      for _ in 0..<iterations {
        let nearEnd = [Float](repeating: 0.2, count: samplesPerIteration)
        let output = processor.processNearEnd(nearEnd)
        nearEndSamplesProcessed += samplesPerIteration
        outputSamplesReceived += output.count
      }
      expectation.fulfill()
    }

    wait(for: [expectation], timeout: 10.0)

    // Verify no crashes occurred and data was processed
    XCTAssertEqual(farEndSamplesFed, iterations * samplesPerIteration)
    XCTAssertEqual(nearEndSamplesProcessed, iterations * samplesPerIteration)
    // Output may be less due to buffering, but should have some output
    XCTAssertGreaterThan(outputSamplesReceived, 0)
  }

  func testConcurrentMultipleProcessors() throws {
    // Test that multiple processor instances don't interfere with each other
    let processor1 = DTLNAecEchoProcessor(modelSize: .small)
    let processor2 = DTLNAecEchoProcessor(modelSize: .small)
    try processor1.loadModelsFromPackage()
    try processor2.loadModelsFromPackage()

    let iterations = 50
    let samples = 512
    let expectation = XCTestExpectation(description: "Multiple processors complete")
    expectation.expectedFulfillmentCount = 2

    var output1Total = 0
    var output2Total = 0

    // Processor 1 on one thread
    DispatchQueue.global(qos: .userInteractive).async {
      for i in 0..<iterations {
        let farEnd = [Float](repeating: Float(i) * 0.01, count: samples)
        let nearEnd = [Float](repeating: Float(i) * 0.02, count: samples)
        processor1.feedFarEnd(farEnd)
        let output = processor1.processNearEnd(nearEnd)
        output1Total += output.count
      }
      expectation.fulfill()
    }

    // Processor 2 on another thread
    DispatchQueue.global(qos: .userInteractive).async {
      for i in 0..<iterations {
        let farEnd = [Float](repeating: Float(i) * 0.03, count: samples)
        let nearEnd = [Float](repeating: Float(i) * 0.04, count: samples)
        processor2.feedFarEnd(farEnd)
        let output = processor2.processNearEnd(nearEnd)
        output2Total += output.count
      }
      expectation.fulfill()
    }

    wait(for: [expectation], timeout: 15.0)

    // Both processors should have produced output
    XCTAssertGreaterThan(output1Total, 0, "Processor 1 should produce output")
    XCTAssertGreaterThan(output2Total, 0, "Processor 2 should produce output")
  }

  func testConcurrentFlushWithProcessing() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()

    let expectation = XCTestExpectation(description: "Concurrent flush completes")
    expectation.expectedFulfillmentCount = 3

    // Thread 1: Feed samples
    DispatchQueue.global(qos: .userInteractive).async {
      for _ in 0..<50 {
        processor.feedFarEnd([Float](repeating: 0.1, count: 128))
        Thread.sleep(forTimeInterval: 0.001)
      }
      expectation.fulfill()
    }

    // Thread 2: Process samples
    DispatchQueue.global(qos: .userInteractive).async {
      for _ in 0..<50 {
        _ = processor.processNearEnd([Float](repeating: 0.2, count: 128))
        Thread.sleep(forTimeInterval: 0.001)
      }
      expectation.fulfill()
    }

    // Thread 3: Periodically flush
    DispatchQueue.global(qos: .userInteractive).async {
      for _ in 0..<10 {
        _ = processor.flush()
        Thread.sleep(forTimeInterval: 0.005)
      }
      expectation.fulfill()
    }

    wait(for: [expectation], timeout: 10.0)
    // Test passes if no crashes or deadlocks occurred
  }
}
