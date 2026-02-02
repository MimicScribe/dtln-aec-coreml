// DTLNAecEchoProcessor.swift
// DTLN-aec Neural Echo Cancellation for CoreML
//
// Two-part neural network for acoustic echo cancellation:
// Part 1: Frequency domain - generates mask from mic + loopback magnitudes
// Part 2: Time domain - refines output using encoded features
//
// Model specs:
// - Sample rate: 16kHz mono
// - Block size: 512 samples (32ms)
// - Block shift: 128 samples (8ms)
// - FFT bins: 257
// - LSTM states: [1, 2, units, 2] (2 layers, N units, h/c states)
//
// Available model sizes:
// - 128 units: 1.8M params, <1ms processing
// - 256 units: 3.9M params, <2ms processing
// - 512 units: 10.4M params, <3ms processing

import Accelerate
import CoreML
import Foundation
import os.log

private let logger = Logger(subsystem: "DTLNAecCoreML", category: "EchoProcessor")

// MARK: - Configuration

/// Configuration options for DTLN-aec echo processor.
///
/// Use this struct to customize model loading and processing behavior.
///
/// ## Example
/// ```swift
/// var config = DTLNAecConfig()
/// config.modelSize = .large
/// config.computeUnits = .cpuAndNeuralEngine
/// config.enablePerformanceTracking = true
///
/// let processor = DTLNAecEchoProcessor(config: config)
/// try await processor.loadModelsAsync()
/// ```
public struct DTLNAecConfig: Sendable {
  /// The model size to use (default: .small for best latency)
  public var modelSize: DTLNAecModelSize

  /// CoreML compute units to use for inference (default: .cpuAndNeuralEngine)
  public var computeUnits: MLComputeUnits

  /// Whether to track performance metrics like average frame time (default: true)
  public var enablePerformanceTracking: Bool

  /// Whether to validate model outputs for NaN/Inf values (default: true)
  /// When enabled, NaN/Inf outputs cause passthrough behavior for that frame.
  public var validateNumerics: Bool

  /// Whether to clip output samples to [-1, 1] range (default: true)
  public var clipOutput: Bool

  /// Creates a configuration with default settings.
  public init(
    modelSize: DTLNAecModelSize = .small,
    computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
    enablePerformanceTracking: Bool = true,
    validateNumerics: Bool = true,
    clipOutput: Bool = true
  ) {
    self.modelSize = modelSize
    self.computeUnits = computeUnits
    self.enablePerformanceTracking = enablePerformanceTracking
    self.validateNumerics = validateNumerics
    self.clipOutput = clipOutput
  }
}

// MARK: - Model Size

/// Available DTLN-aec model sizes.
/// Larger models have better quality but higher latency.
public enum DTLNAecModelSize: Int, CaseIterable, Sendable {
  case small = 128  // 1.8M params, <1ms processing
  case medium = 256  // 3.9M params, <2ms processing
  case large = 512  // 10.4M params, <3ms processing

  public var units: Int { rawValue }

  var modelNamePrefix: String { "DTLN_AEC_\(rawValue)" }

  /// Estimated model file size (both parts combined) in MB
  public var estimatedSizeMB: Double {
    switch self {
    case .small: return 3.6
    case .medium: return 8.0
    case .large: return 20.3
    }
  }
}

/// Neural echo cancellation using DTLN-aec architecture.
/// Processes 8ms frames (128 samples) with overlap-add output.
///
/// ## Usage
/// ```swift
/// let processor = DTLNAecEchoProcessor(modelSize: .small)
/// try processor.loadModels()
///
/// // During audio processing:
/// processor.feedFarEnd(systemAudioSamples)  // 16kHz Float array
/// let cleanAudio = processor.processNearEnd(microphoneSamples)
/// ```
///
/// ## Thread Safety
/// NOT thread-safe. Call from a single thread or serial queue.
public final class DTLNAecEchoProcessor {

  // MARK: - Constants

  /// Audio sample rate (must be 16kHz)
  public static let sampleRate: Double = 16_000

  /// FFT/IFFT block size in samples (512 = 32ms at 16kHz)
  public static let blockLen = 512

  /// Frame shift in samples (128 = 8ms at 16kHz)
  public static let blockShift = 128

  /// Number of FFT bins (blockLen/2 + 1)
  public static let fftBins = 257

  /// log2(blockLen) for FFT setup
  private static let fftLog2n = vDSP_Length(Int(log2(Double(blockLen))))

  /// Number of LSTM layers
  static let numLayers = 2

  // MARK: - Model Configuration

  /// The configuration used by this processor
  public let config: DTLNAecConfig

  /// The model size being used
  public var modelSize: DTLNAecModelSize { config.modelSize }

  /// Number of LSTM units per layer (from model size)
  public var numUnits: Int { config.modelSize.units }

  // MARK: - CoreML Models

  private var modelPart1: MLModel?
  private var modelPart2: MLModel?
  private var modelBundle: Bundle?

  // MARK: - LSTM States (persist across frames)

  private var states1: MLMultiArray?
  private var states2: MLMultiArray?

  // MARK: - Audio Buffers
  // Python-style: Fixed-size sliding windows, pre-initialized with zeros
  // This provides smooth startup matching the TFLite reference behavior

  private var micBuffer: [Float]
  private var loopbackBuffer: [Float]
  private var outputBuffer: [Float]

  // MARK: - FFT Setup (Accelerate vDSP)

  private var fftSetup: OpaquePointer?
  private var window: [Float]
  private var fftRealBuffer: [Float]  // Size: blockLen/2 for packed real FFT
  private var fftImagBuffer: [Float]  // Size: blockLen/2 for packed real FFT
  private var magnitudeSquaredBuffer: [Float]  // Size: fftBins for vDSP magnitude calculation

  // MARK: - Preallocated MLMultiArrays

  private var micMagArray: MLMultiArray?
  private var lpbMagArray: MLMultiArray?
  private var estimatedFrameArray: MLMultiArray?
  private var lpbTimeArray: MLMultiArray?

  // MARK: - Statistics

  private var framesProcessed: Int = 0
  private var totalProcessingTimeMs: Double = 0

  // MARK: - Initialization

  /// Whether models are loaded and ready for processing
  public var isInitialized: Bool {
    modelPart1 != nil && modelPart2 != nil
  }

  /// Initialize with specified configuration.
  /// - Parameter config: The configuration for this processor
  public init(config: DTLNAecConfig) {
    self.config = config
    // Python-style: Pre-fill buffers with zeros for smooth startup
    micBuffer = [Float](repeating: 0, count: Self.blockLen)
    loopbackBuffer = [Float](repeating: 0, count: Self.blockLen)
    outputBuffer = [Float](repeating: 0, count: Self.blockLen)
    window = [Float](repeating: 0, count: Self.blockLen)
    vDSP_hann_window(&window, vDSP_Length(Self.blockLen), Int32(vDSP_HANN_NORM))
    // For packed real FFT, buffers are half the block size
    fftRealBuffer = [Float](repeating: 0, count: Self.blockLen / 2)
    fftImagBuffer = [Float](repeating: 0, count: Self.blockLen / 2)
    // For vDSP magnitude calculation
    magnitudeSquaredBuffer = [Float](repeating: 0, count: Self.fftBins)
    // Ring buffers for pending samples (O(1) operations)
    pendingMicRing = RingBuffer(capacity: Self.ringBufferCapacity)
    pendingLoopbackRing = RingBuffer(capacity: Self.ringBufferCapacity)
    // log2n for FFT setup (computed from blockLen)
    fftSetup = vDSP_create_fftsetup(Self.fftLog2n, FFTRadix(kFFTRadix2))
  }

  deinit {
    if let fftSetup {
      vDSP_destroy_fftsetup(fftSetup)
    }
  }

  /// Initialize with specified model size using default configuration.
  /// - Parameter modelSize: The DTLN-aec model variant to use (default: .small = 128 units)
  public convenience init(modelSize: DTLNAecModelSize = .small) {
    self.init(config: DTLNAecConfig(modelSize: modelSize))
  }

  /// Load CoreML models from bundle.
  /// Call this before processing audio.
  /// - Parameter bundle: The bundle containing model resources. If nil, searches module and main bundles.
  /// - Throws: `DTLNAecError.modelNotFound` if models are not in the bundle
  public func loadModels(from bundle: Bundle? = nil) throws {
    self.modelBundle = bundle
    let startTime = Date()

    let part1Name = "\(modelSize.modelNamePrefix)_Part1"
    let part2Name = "\(modelSize.modelNamePrefix)_Part2"

    guard let part1URL = try findAndCompileModel(name: part1Name) else {
      throw DTLNAecError.modelNotFound(part1Name)
    }

    guard let part2URL = try findAndCompileModel(name: part2Name) else {
      throw DTLNAecError.modelNotFound(part2Name)
    }

    let mlConfig = MLModelConfiguration()
    mlConfig.computeUnits = config.computeUnits

    modelPart1 = try MLModel(contentsOf: part1URL, configuration: mlConfig)
    modelPart2 = try MLModel(contentsOf: part2URL, configuration: mlConfig)

    try initializeStates()
    try preallocateArrays()

    let loadTimeMs = Date().timeIntervalSince(startTime) * 1000
    logger.info(
      "DTLN-aec \(self.modelSize.units)-unit models loaded in \(String(format: "%.1f", loadTimeMs))ms"
    )
  }

  /// Asynchronously load CoreML models from bundle.
  /// This performs model compilation on a background thread to avoid blocking the main thread.
  /// - Parameter bundle: The bundle containing model resources. If nil, searches module and main bundles.
  /// - Throws: `DTLNAecError.modelNotFound` if models are not in the bundle
  @available(macOS 10.15, iOS 13.0, *)
  public func loadModelsAsync(from bundle: Bundle? = nil) async throws {
    try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
      DispatchQueue.global(qos: .userInitiated).async {
        do {
          try self.loadModels(from: bundle)
          continuation.resume()
        } catch {
          continuation.resume(throwing: error)
        }
      }
    }
  }

  private func findAndCompileModel(name: String) throws -> URL? {
    // Check provided model bundle first (for split model packages)
    if let bundle = modelBundle {
      if let url = bundle.url(forResource: name, withExtension: "mlmodelc") {
        return url
      }
      if let url = bundle.url(forResource: name, withExtension: "mlpackage") {
        logger.info("Compiling \(name).mlpackage...")
        return try MLModel.compileModel(at: url)
      }
    }

    // Check for pre-compiled model in main bundle
    if let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") {
      return url
    }

    // Check main bundle for mlpackage
    if let url = Bundle.main.url(forResource: name, withExtension: "mlpackage") {
      logger.info("Compiling \(name).mlpackage...")
      return try MLModel.compileModel(at: url)
    }

    return nil
  }

  private func initializeStates() throws {
    let stateShape = [1, Self.numLayers, numUnits, 2] as [NSNumber]
    // Try float32 for all models and see if there's a type mismatch
    let stateType: MLMultiArrayDataType = .float32
    states1 = try MLMultiArray(shape: stateShape, dataType: stateType)
    states2 = try MLMultiArray(shape: stateShape, dataType: stateType)
    resetStates()
  }

  private func preallocateArrays() throws {
    let magShape = [1, 1, Self.fftBins] as [NSNumber]
    micMagArray = try MLMultiArray(shape: magShape, dataType: .float32)
    lpbMagArray = try MLMultiArray(shape: magShape, dataType: .float32)

    let timeShape = [1, 1, Self.blockLen] as [NSNumber]
    estimatedFrameArray = try MLMultiArray(shape: timeShape, dataType: .float32)
    lpbTimeArray = try MLMultiArray(shape: timeShape, dataType: .float32)
  }

  /// Reset LSTM states to zeros (call when starting new recording)
  public func resetStates() {
    guard let states1, let states2 else { return }

    let count = states1.count
    if states1.dataType == .float16 {
      let ptr1 = states1.dataPointer.assumingMemoryBound(to: Float16.self)
      let ptr2 = states2.dataPointer.assumingMemoryBound(to: Float16.self)
      for i in 0..<count {
        ptr1[i] = 0
        ptr2[i] = 0
      }
    } else {
      let ptr1 = states1.dataPointer.assumingMemoryBound(to: Float.self)
      let ptr2 = states2.dataPointer.assumingMemoryBound(to: Float.self)
      for i in 0..<count {
        ptr1[i] = 0
        ptr2[i] = 0
      }
    }

    // Python-style: Reset to fixed-size zero-filled buffers
    micBuffer = [Float](repeating: 0, count: Self.blockLen)
    loopbackBuffer = [Float](repeating: 0, count: Self.blockLen)
    outputBuffer = [Float](repeating: 0, count: Self.blockLen)
    pendingMicSamples.removeAll(keepingCapacity: true)
    pendingLoopbackSamples.removeAll(keepingCapacity: true)
    framesProcessed = 0
    totalProcessingTimeMs = 0
  }

  // MARK: - Pending Sample Ring Buffers (O(1) append/remove)

  /// Fixed-capacity ring buffer for O(1) sample operations
  private struct RingBuffer {
    private var storage: [Float]
    private var readIndex: Int = 0
    private var writeIndex: Int = 0
    private(set) var count: Int = 0
    let capacity: Int

    init(capacity: Int) {
      self.capacity = capacity
      self.storage = [Float](repeating: 0, count: capacity)
    }

    mutating func append(contentsOf samples: [Float]) {
      for sample in samples {
        storage[writeIndex] = sample
        writeIndex = (writeIndex + 1) % capacity
        if count < capacity {
          count += 1
        } else {
          // Overwrite oldest - move read index
          readIndex = (readIndex + 1) % capacity
        }
      }
    }

    mutating func removeFirst(_ n: Int) {
      let removeCount = min(n, count)
      readIndex = (readIndex + removeCount) % capacity
      count -= removeCount
    }

    func prefix(_ n: Int) -> [Float] {
      let takeCount = min(n, count)
      var result = [Float](repeating: 0, count: takeCount)
      for i in 0..<takeCount {
        result[i] = storage[(readIndex + i) % capacity]
      }
      return result
    }

    var isEmpty: Bool { count == 0 }

    mutating func removeAll(keepingCapacity: Bool = false) {
      readIndex = 0
      writeIndex = 0
      count = 0
    }
  }

  // Ring buffers with 4096 sample capacity (~256ms at 16kHz)
  private static let ringBufferCapacity = 4096
  private var pendingMicRing: RingBuffer
  private var pendingLoopbackRing: RingBuffer

  // Legacy accessors for compatibility
  private var pendingMicSamples: RingBuffer {
    get { pendingMicRing }
    set { pendingMicRing = newValue }
  }
  private var pendingLoopbackSamples: RingBuffer {
    get { pendingLoopbackRing }
    set { pendingLoopbackRing = newValue }
  }

  // MARK: - Public API

  /// Feed far-end (loopback/system audio) samples.
  /// Call this BEFORE processNearEnd for proper echo reference.
  /// - Parameter samples: Audio samples at 16kHz, Float format
  public func feedFarEnd(_ samples: [Float]) {
    pendingLoopbackSamples.append(contentsOf: samples)
  }

  /// Process near-end (microphone) samples and return echo-cancelled output.
  /// Returns the same number of samples as input (with processing delay).
  /// - Parameter samples: Microphone audio at 16kHz, Float format
  /// - Returns: Echo-cancelled audio samples
  public func processNearEnd(_ samples: [Float]) -> [Float] {
    guard isInitialized else {
      logger.warning("DTLN-aec not initialized, passing through")
      return samples
    }

    pendingMicSamples.append(contentsOf: samples)
    var outputSamples: [Float] = []

    // Process in blockShift (128 sample) chunks - Python style
    while pendingMicSamples.count >= Self.blockShift
      && pendingLoopbackSamples.count >= Self.blockShift
    {
      let frameStart = Date()

      // Python-style sliding window: shift left and add new samples at end
      // micBuffer and loopbackBuffer are always exactly blockLen (512) samples
      shiftAndAppend(
        buffer: &micBuffer, newSamples: Array(pendingMicSamples.prefix(Self.blockShift)))
      shiftAndAppend(
        buffer: &loopbackBuffer, newSamples: Array(pendingLoopbackSamples.prefix(Self.blockShift)))

      pendingMicSamples.removeFirst(Self.blockShift)
      pendingLoopbackSamples.removeFirst(Self.blockShift)

      // Process the current 512-sample window
      if let processed = processFrame(mic: micBuffer, loopback: loopbackBuffer) {
        let frameOutput = overlapAdd(processed)
        outputSamples.append(contentsOf: frameOutput)
      } else {
        let frameOutput = overlapAdd(micBuffer)
        outputSamples.append(contentsOf: frameOutput)
      }

      if config.enablePerformanceTracking {
        framesProcessed += 1
        totalProcessingTimeMs += Date().timeIntervalSince(frameStart) * 1000
      }
    }

    return outputSamples
  }

  /// Number of samples buffered but not yet output.
  /// This includes pending samples waiting to form a complete frame (0-127 samples)
  /// plus the overlap-add tail (384 samples).
  public var pendingSampleCount: Int {
    let overlapTail = Self.blockLen - Self.blockShift  // 384
    return pendingMicSamples.count + overlapTail
  }

  /// Flush remaining buffered audio at end of recording.
  ///
  /// When a recording ends, there may be samples that haven't been output yet:
  /// - Pending samples (0-127) waiting for enough samples to form a frame
  /// - Overlap-add tail (384 samples) from previous frame processing
  ///
  /// This method processes any pending samples (zero-padded if needed) and returns
  /// all remaining buffered audio. Total output is up to 511 samples (~32ms at 16kHz).
  ///
  /// - Note: LSTM states are NOT reset by this method, preserving session continuity.
  ///   Call `resetStates()` separately if starting a new recording session.
  ///
  /// - Returns: Remaining buffered audio samples (pending + overlap-add tail)
  public func flush() -> [Float] {
    guard isInitialized else {
      let samples = pendingMicSamples.prefix(pendingMicSamples.count)
      pendingMicSamples.removeAll(keepingCapacity: true)
      pendingLoopbackSamples.removeAll(keepingCapacity: true)
      return samples
    }

    var outputSamples: [Float] = []

    // Process any pending samples by zero-padding to blockShift boundary
    if !pendingMicSamples.isEmpty || !pendingLoopbackSamples.isEmpty {
      // Pad mic samples to blockShift
      let micPadCount = Self.blockShift - pendingMicSamples.count
      if micPadCount > 0 {
        pendingMicSamples.append(contentsOf: [Float](repeating: 0, count: micPadCount))
      }

      // Pad loopback samples to blockShift
      let lpbPadCount = Self.blockShift - pendingLoopbackSamples.count
      if lpbPadCount > 0 {
        pendingLoopbackSamples.append(contentsOf: [Float](repeating: 0, count: lpbPadCount))
      }

      // Process the padded frame through normal path
      shiftAndAppend(
        buffer: &micBuffer, newSamples: Array(pendingMicSamples.prefix(Self.blockShift)))
      shiftAndAppend(
        buffer: &loopbackBuffer, newSamples: Array(pendingLoopbackSamples.prefix(Self.blockShift)))

      pendingMicSamples.removeAll(keepingCapacity: true)
      pendingLoopbackSamples.removeAll(keepingCapacity: true)

      if let processed = processFrame(mic: micBuffer, loopback: loopbackBuffer) {
        let frameOutput = overlapAdd(processed)
        outputSamples.append(contentsOf: frameOutput)
      } else {
        let frameOutput = overlapAdd(micBuffer)
        outputSamples.append(contentsOf: frameOutput)
      }
    }

    // Extract the overlap-add tail (first 384 samples that haven't been output yet)
    let overlapTail = Self.blockLen - Self.blockShift
    outputSamples.append(contentsOf: outputBuffer.prefix(overlapTail))

    // Zero the output buffer (but don't reset LSTM states)
    for i in 0..<Self.blockLen {
      outputBuffer[i] = 0
    }

    return outputSamples
  }

  /// Python-style buffer shift: shift left by blockShift and add new samples at end
  private func shiftAndAppend(buffer: inout [Float], newSamples: [Float]) {
    let overlapCount = Self.blockLen - Self.blockShift  // 384
    // Use memmove for overlapping memory regions (source and dest overlap)
    _ = buffer.withUnsafeMutableBytes { ptr in
      memmove(
        ptr.baseAddress!,
        ptr.baseAddress! + Self.blockShift * MemoryLayout<Float>.stride,
        overlapCount * MemoryLayout<Float>.stride)
    }
    // Add new samples at end (last 128 positions)
    let copyCount = min(newSamples.count, Self.blockShift)
    if copyCount > 0 {
      newSamples.withUnsafeBufferPointer { srcPtr in
        buffer.withUnsafeMutableBufferPointer { dstPtr in
          dstPtr.baseAddress!.advanced(by: overlapCount)
            .update(from: srcPtr.baseAddress!, count: copyCount)
        }
      }
    }
    // Zero-fill if newSamples is shorter than blockShift
    if copyCount < Self.blockShift {
      buffer.withUnsafeMutableBufferPointer { ptr in
        ptr.baseAddress!.advanced(by: overlapCount + copyCount)
          .update(repeating: 0, count: Self.blockShift - copyCount)
      }
    }
  }

  /// Get average frame processing time in milliseconds
  public var averageFrameTimeMs: Double {
    guard framesProcessed > 0 else { return 0 }
    return totalProcessingTimeMs / Double(framesProcessed)
  }

  // MARK: - Frame Processing

  private func processFrame(mic: [Float], loopback: [Float]) -> [Float]? {
    guard let modelPart1, let modelPart2,
      let states1, let states2,
      let micMagArray, let lpbMagArray,
      let estimatedFrameArray, let lpbTimeArray
    else { return nil }

    // Part 1: Frequency Domain
    let (micMag, micPhase) = computeMagnitudeAndPhase(mic)
    let (lpbMag, _) = computeMagnitudeAndPhase(loopback)

    copyToMLArray(micMag, to: micMagArray)
    copyToMLArray(lpbMag, to: lpbMagArray)

    let part1Input: [String: Any] = [
      "mic_magnitude": micMagArray,
      "lpb_magnitude": lpbMagArray,
      "states_in": states1,
    ]

    guard let part1Provider = try? MLDictionaryFeatureProvider(dictionary: part1Input),
      let part1Output = try? modelPart1.prediction(from: part1Provider),
      let mask = part1Output.featureValue(for: "Identity")?.multiArrayValue,
      let newStates1 = part1Output.featureValue(for: "Identity_1")?.multiArrayValue
    else { return nil }

    copyStates(from: newStates1, to: states1)
    let estimatedFrame = applyMaskAndIFFT(micSamples: mic, micPhase: micPhase, mask: mask)

    // Part 2: Time Domain
    copyToMLArray(estimatedFrame, to: estimatedFrameArray)
    copyToMLArray(loopback, to: lpbTimeArray)

    let part2Input: [String: Any] = [
      "estimated_frame": estimatedFrameArray,
      "lpb_time": lpbTimeArray,
      "states_in": states2,
    ]

    guard let part2Provider = try? MLDictionaryFeatureProvider(dictionary: part2Input),
      let part2Output = try? modelPart2.prediction(from: part2Provider),
      let output = part2Output.featureValue(for: "Identity")?.multiArrayValue,
      let newStates2 = part2Output.featureValue(for: "Identity_1")?.multiArrayValue
    else { return nil }

    copyStates(from: newStates2, to: states2)
    var result = extractFromMLArray(output, count: Self.blockLen)

    // Validate numerics if enabled
    if config.validateNumerics && containsInvalidNumerics(result) {
      // Fall back to passthrough (mic input) on NaN/Inf
      return mic
    }

    // Clip output if enabled
    if config.clipOutput {
      clipToValidRange(&result)
    }

    return result
  }

  // MARK: - FFT Helpers

  private func computeMagnitudeAndPhase(_ samples: [Float]) -> (magnitude: [Float], phase: [Float])
  {
    guard let fftSetup else {
      return (
        [Float](repeating: 0, count: Self.fftBins), [Float](repeating: 0, count: Self.fftBins)
      )
    }

    // Pack real samples for vDSP real FFT using stride-based copy
    let halfLen = Self.blockLen / 2
    samples.withUnsafeBufferPointer { srcPtr in
      fftRealBuffer.withUnsafeMutableBufferPointer { realPtr in
        fftImagBuffer.withUnsafeMutableBufferPointer { imagPtr in
          // Copy even indices to real, odd indices to imag (stride 2)
          vDSP_vsadd(
            srcPtr.baseAddress!, 2, [Float](repeating: 0, count: 1), realPtr.baseAddress!, 1,
            vDSP_Length(halfLen))
          vDSP_vsadd(
            srcPtr.baseAddress! + 1, 2, [Float](repeating: 0, count: 1), imagPtr.baseAddress!, 1,
            vDSP_Length(halfLen))
        }
      }
    }

    var magnitude = [Float](repeating: 0, count: Self.fftBins)
    var phase = [Float](repeating: 0, count: Self.fftBins)

    fftRealBuffer.withUnsafeMutableBufferPointer { realPtr in
      fftImagBuffer.withUnsafeMutableBufferPointer { imagPtr in
        var splitComplex = DSPSplitComplex(realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!)

        // Forward real FFT - output is packed: DC in realp[0], Nyquist in imagp[0]
        vDSP_fft_zrip(fftSetup, &splitComplex, 1, vDSP_Length(9), FFTDirection(FFT_FORWARD))

        // DC bin (index 0) - stored in realp[0], purely real
        magnitude[0] = abs(realPtr[0])
        phase[0] = realPtr[0] >= 0 ? 0 : .pi

        // Nyquist bin (index fftBins-1 = 256) - stored in imagp[0], purely real
        magnitude[Self.fftBins - 1] = abs(imagPtr[0])
        phase[Self.fftBins - 1] = imagPtr[0] >= 0 ? 0 : .pi

        // Bins 1 to fftBins-2 (indices 1..255) - use vDSP for magnitude
        // Create a view of bins 1..255 as a split complex
        var midSplit = DSPSplitComplex(
          realp: realPtr.baseAddress! + 1, imagp: imagPtr.baseAddress! + 1)
        magnitude.withUnsafeMutableBufferPointer { magPtr in
          // Calculate squared magnitudes: real^2 + imag^2
          vDSP_zvmags(&midSplit, 1, magPtr.baseAddress! + 1, 1, vDSP_Length(Self.fftBins - 2))
        }

        // Phase for bins 1..255 still needs atan2
        for i in 1..<(Self.fftBins - 1) {
          phase[i] = atan2(imagPtr[i], realPtr[i])
        }
      }
    }

    // Clamp to zero before sqrt to prevent NaN from floating-point errors
    var lowerBound: Float = 0.0
    var upperBound: Float = .greatestFiniteMagnitude
    magnitude.withUnsafeMutableBufferPointer { magPtr in
      vDSP_vclip(
        magPtr.baseAddress! + 1, 1, &lowerBound, &upperBound,
        magPtr.baseAddress! + 1, 1, vDSP_Length(Self.fftBins - 2))
    }

    // Take sqrt of squared magnitudes for bins 1..255
    var sqrtCount = Int32(Self.fftBins - 2)
    magnitude.withUnsafeMutableBufferPointer { magPtr in
      vvsqrtf(magPtr.baseAddress! + 1, magPtr.baseAddress! + 1, &sqrtCount)
    }

    // vDSP real FFT scales by 2, compensate to match NumPy rfft
    vDSP.multiply(0.5, magnitude, result: &magnitude)
    return (magnitude, phase)
  }

  private func applyMaskAndIFFT(
    micSamples: [Float], micPhase: [Float], mask: MLMultiArray
  )
    -> [Float]
  {
    guard let fftSetup else { return micSamples }

    // Pack real samples for vDSP real FFT using stride-based copy
    let halfLen = Self.blockLen / 2
    micSamples.withUnsafeBufferPointer { srcPtr in
      fftRealBuffer.withUnsafeMutableBufferPointer { realPtr in
        fftImagBuffer.withUnsafeMutableBufferPointer { imagPtr in
          // Copy even indices to real, odd indices to imag (stride 2)
          vDSP_vsadd(
            srcPtr.baseAddress!, 2, [Float](repeating: 0, count: 1), realPtr.baseAddress!, 1,
            vDSP_Length(halfLen))
          vDSP_vsadd(
            srcPtr.baseAddress! + 1, 2, [Float](repeating: 0, count: 1), imagPtr.baseAddress!, 1,
            vDSP_Length(halfLen))
        }
      }
    }

    var output = [Float](repeating: 0, count: Self.blockLen)

    // Helper to get mask value at index
    let getMask: (Int) -> Float = { index in
      if mask.dataType == .float16 {
        let maskPtr = mask.dataPointer.assumingMemoryBound(to: Float16.self)
        return Float(maskPtr[index])
      } else {
        let maskPtr = mask.dataPointer.assumingMemoryBound(to: Float.self)
        return maskPtr[index]
      }
    }

    fftRealBuffer.withUnsafeMutableBufferPointer { realPtr in
      fftImagBuffer.withUnsafeMutableBufferPointer { imagPtr in
        var splitComplex = DSPSplitComplex(realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!)

        // Forward real FFT
        vDSP_fft_zrip(fftSetup, &splitComplex, 1, vDSP_Length(9), FFTDirection(FFT_FORWARD))

        // Apply mask with proper handling of packed DC/Nyquist
        // DC bin (index 0): realp[0] contains DC, apply mask[0]
        realPtr[0] *= getMask(0)

        // Nyquist bin: imagp[0] contains Nyquist, apply mask[fftBins-1] (index 256)
        imagPtr[0] *= getMask(Self.fftBins - 1)

        // Regular bins 1 to fftBins-2 (indices 1..255)
        for i in 1..<(Self.fftBins - 1) {
          let m = getMask(i)
          realPtr[i] *= m
          imagPtr[i] *= m
        }

        // Inverse real FFT - vDSP handles conjugate symmetry internally
        vDSP_fft_zrip(fftSetup, &splitComplex, 1, vDSP_Length(9), FFTDirection(FFT_INVERSE))

        // Unpack using stride-based copy: output[2*i] = realp[i], output[2*i+1] = imagp[i]
        output.withUnsafeMutableBufferPointer { outPtr in
          vDSP_vsadd(
            realPtr.baseAddress!, 1, [Float](repeating: 0, count: 1), outPtr.baseAddress!, 2,
            vDSP_Length(halfLen))
          vDSP_vsadd(
            imagPtr.baseAddress!, 1, [Float](repeating: 0, count: 1), outPtr.baseAddress! + 1, 2,
            vDSP_Length(halfLen))
        }
      }
    }

    // vDSP real FFT scales by 2 on forward and 2 on inverse = 4x total
    // Divide by 2*N to get correct amplitude
    vDSP.multiply(1.0 / Float(2 * Self.blockLen), output, result: &output)
    return output
  }

  // MARK: - Overlap-Add

  private func overlapAdd(_ frame: [Float]) -> [Float] {
    vDSP.add(outputBuffer, frame, result: &outputBuffer)
    let output = Array(outputBuffer.prefix(Self.blockShift))

    let overlapCount = Self.blockLen - Self.blockShift  // 384
    // Use memmove-style bulk copy instead of element-by-element loop
    outputBuffer.withUnsafeMutableBufferPointer { ptr in
      ptr.baseAddress!.update(from: ptr.baseAddress! + Self.blockShift, count: overlapCount)
      // Zero the tail
      ptr.baseAddress!.advanced(by: overlapCount).update(repeating: 0, count: Self.blockShift)
    }

    return output
  }

  // MARK: - MLMultiArray Helpers

  private func copyToMLArray(_ source: [Float], to array: MLMultiArray) {
    if array.dataType == .float16 {
      let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
      for i in 0..<min(source.count, array.count) {
        ptr[i] = Float16(source[i])
      }
    } else {
      let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
      for i in 0..<min(source.count, array.count) {
        ptr[i] = source[i]
      }
    }
  }

  private func extractFromMLArray(_ array: MLMultiArray, count: Int) -> [Float] {
    var result = [Float](repeating: 0, count: count)
    if array.dataType == .float16 {
      let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
      for i in 0..<min(count, array.count) {
        result[i] = Float(ptr[i])
      }
    } else {
      let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
      for i in 0..<min(count, array.count) {
        result[i] = ptr[i]
      }
    }
    return result
  }

  private func copyStates(from source: MLMultiArray, to dest: MLMultiArray) {
    let count = min(source.count, dest.count)
    if source.dataType == .float16 && dest.dataType == .float16 {
      // Both float16 - direct copy
      let srcPtr = source.dataPointer.assumingMemoryBound(to: Float16.self)
      let dstPtr = dest.dataPointer.assumingMemoryBound(to: Float16.self)
      for i in 0..<count {
        dstPtr[i] = srcPtr[i]
      }
    } else if source.dataType == .float16 {
      // float16 -> float32
      let srcPtr = source.dataPointer.assumingMemoryBound(to: Float16.self)
      let dstPtr = dest.dataPointer.assumingMemoryBound(to: Float.self)
      for i in 0..<count {
        dstPtr[i] = Float(srcPtr[i])
      }
    } else {
      // float32 -> float32
      let srcPtr = source.dataPointer.assumingMemoryBound(to: Float.self)
      let dstPtr = dest.dataPointer.assumingMemoryBound(to: Float.self)
      for i in 0..<count {
        dstPtr[i] = srcPtr[i]
      }
    }
  }

  // MARK: - Numeric Validation

  /// Check if array contains any NaN or Inf values
  private func containsInvalidNumerics(_ array: [Float]) -> Bool {
    for value in array {
      if value.isNaN || value.isInfinite {
        return true
      }
    }
    return false
  }

  /// Clip array values to [-1, 1] range in place
  private func clipToValidRange(_ array: inout [Float]) {
    let count = array.count
    var lowerBound: Float = -1.0
    var upperBound: Float = 1.0
    array.withUnsafeMutableBufferPointer { ptr in
      vDSP_vclip(
        ptr.baseAddress!, 1, &lowerBound, &upperBound,
        ptr.baseAddress!, 1, vDSP_Length(count))
    }
  }
}

// MARK: - Errors

/// Errors that can occur during DTLN-aec processing
public enum DTLNAecError: Error, LocalizedError {
  case modelNotFound(String)
  case initializationFailed(String)
  case inferenceFailed(String)

  public var errorDescription: String? {
    switch self {
    case .modelNotFound(let name):
      return
        "DTLN-aec model not found: \(name). Ensure the mlpackage files are included in your bundle."
    case .initializationFailed(let reason):
      return "DTLN-aec initialization failed: \(reason)"
    case .inferenceFailed(let reason):
      return "DTLN-aec inference failed: \(reason)"
    }
  }
}
