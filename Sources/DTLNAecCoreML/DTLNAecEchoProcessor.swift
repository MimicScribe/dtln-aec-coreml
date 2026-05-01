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
import os
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

  /// Whether to apply linked gain control to normalize input levels (default: true)
  ///
  /// When enabled, applies the same gain to both mic and system audio to preserve
  /// the echo-to-reference ratio needed for stable DTLN convergence. Uses soft-knee
  /// compression to handle loud transients without clipping.
  public var enableLinkedGainControl: Bool

  /// Creates a configuration with default settings.
  public init(
    modelSize: DTLNAecModelSize = .small,
    computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
    enablePerformanceTracking: Bool = true,
    validateNumerics: Bool = true,
    clipOutput: Bool = true,
    enableLinkedGainControl: Bool = true
  ) {
    self.modelSize = modelSize
    self.computeUnits = computeUnits
    self.enablePerformanceTracking = enablePerformanceTracking
    self.validateNumerics = validateNumerics
    self.clipOutput = clipOutput
    self.enableLinkedGainControl = enableLinkedGainControl
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
/// Thread-safe. `feedFarEnd`, `processNearEnd`, and `flush` are protected by an internal lock.
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
  private var effectiveMicBuffer: [Float]
  private var effectiveLoopbackBuffer: [Float]
  private var micBufferPeak: Float = 0
  private var loopbackBufferPeak: Float = 0

  // MARK: - FFT Setup (Accelerate vDSP)

  private var fftSetup: OpaquePointer?
  private var window: [Float]
  // Scratch split-complex for the loopback forward FFT (magnitude only).
  // Size: blockLen/2 for vDSP packed real FFT.
  private var fftRealBuffer: [Float]
  private var fftImagBuffer: [Float]
  // Persistent split-complex for the mic forward FFT. Held across the Part1
  // model call so applyMaskAndIFFT can reuse it for the inverse pass instead
  // of redoing the forward FFT on the same samples.
  private var micFftRealBuffer: [Float]
  private var micFftImagBuffer: [Float]

  // Per-frame scratch buffers, preallocated once and reused. Without these,
  // each frame freed/allocated ~10 transient [Float]s on the hot path.
  private var micMagBuffer: [Float]  // size: fftBins (mic magnitude spectrum)
  private var lpbMagBuffer: [Float]  // size: fftBins (loopback magnitude spectrum)
  private var maskScratch: [Float]  // size: fftBins (mask extracted from MLMultiArray as fp32)
  private var frameOutputBuffer: [Float]  // size: blockLen (IFFT time-domain output)
  private var frameResultBuffer: [Float]  // size: blockLen (Part2 model output, time domain)
  private var prefixReadBuffer: [Float]  // size: blockShift (ring prefix scratch)

  // MARK: - Preallocated MLMultiArrays

  private var micMagArray: MLMultiArray?
  private var lpbMagArray: MLMultiArray?
  private var estimatedFrameArray: MLMultiArray?
  private var lpbTimeArray: MLMultiArray?

  // MARK: - Statistics

  private var framesProcessed: Int = 0
  private var totalProcessingTimeMs: Double = 0

  // MARK: - Linked Gain Control

  private var gainController: LinkedGainController?

  // MARK: - Thread Safety

  /// Lock protecting feedFarEnd, processNearEnd, and flush for concurrent access
  private var processingLock = os_unfair_lock()

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
    effectiveMicBuffer = [Float](repeating: 0, count: Self.blockLen)
    effectiveLoopbackBuffer = [Float](repeating: 0, count: Self.blockLen)
    window = [Float](repeating: 0, count: Self.blockLen)
    vDSP_hann_window(&window, vDSP_Length(Self.blockLen), Int32(vDSP_HANN_NORM))
    // For packed real FFT, buffers are half the block size
    fftRealBuffer = [Float](repeating: 0, count: Self.blockLen / 2)
    fftImagBuffer = [Float](repeating: 0, count: Self.blockLen / 2)
    micFftRealBuffer = [Float](repeating: 0, count: Self.blockLen / 2)
    micFftImagBuffer = [Float](repeating: 0, count: Self.blockLen / 2)
    // Per-frame scratch (avoid hot-path allocations)
    micMagBuffer = [Float](repeating: 0, count: Self.fftBins)
    lpbMagBuffer = [Float](repeating: 0, count: Self.fftBins)
    maskScratch = [Float](repeating: 0, count: Self.fftBins)
    frameOutputBuffer = [Float](repeating: 0, count: Self.blockLen)
    frameResultBuffer = [Float](repeating: 0, count: Self.blockLen)
    prefixReadBuffer = [Float](repeating: 0, count: Self.blockShift)
    // Ring buffers for pending samples (O(1) operations)
    pendingMicRing = RingBuffer(capacity: Self.ringBufferCapacity)
    pendingLoopbackRing = RingBuffer(capacity: Self.ringBufferCapacity)
    // log2n for FFT setup (computed from blockLen)
    fftSetup = vDSP_create_fftsetup(Self.fftLog2n, FFTRadix(kFFTRadix2))
    // Initialize linked gain control if enabled
    if config.enableLinkedGainControl {
      gainController = LinkedGainController()
    }
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
    // fp32 matches the model's `states_in` port. fp16 was A/B'd (RMS ratio
    // held at 1.02x) but regressed P99 by 9-24% across all 3 model sizes on
    // the ANE — the model's state input is fp32-native, so fp16 adds an
    // inbound conversion that outweighs the half-size copyStates memcpy.
    states1 = try MLMultiArray(shape: stateShape, dataType: .float32)
    states2 = try MLMultiArray(shape: stateShape, dataType: .float32)
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
    pendingMicRing.removeAll(keepingCapacity: true)
    pendingLoopbackRing.removeAll(keepingCapacity: true)
    framesProcessed = 0
    totalProcessingTimeMs = 0
    micBufferPeak = 0
    loopbackBufferPeak = 0
    gainController?.reset()
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

    /// Copy up to `n` samples from the head of the ring into `dest` (which
    /// must have at least `n` elements). Avoids the fresh `[Float]`
    /// allocation that `prefix(_:)` makes — meaningful on the per-frame
    /// audio path. Returns the number actually copied.
    @discardableResult
    func copyPrefix(into dest: inout [Float], count n: Int) -> Int {
      let takeCount = min(n, count)
      // Fast path: contiguous (no ring wrap).
      let endIndex = readIndex + takeCount
      if endIndex <= capacity {
        storage.withUnsafeBufferPointer { srcPtr in
          dest.withUnsafeMutableBufferPointer { dstPtr in
            dstPtr.baseAddress!.update(
              from: srcPtr.baseAddress! + readIndex, count: takeCount)
          }
        }
      } else {
        // Wrap path: split into two memcpys.
        let first = capacity - readIndex
        storage.withUnsafeBufferPointer { srcPtr in
          dest.withUnsafeMutableBufferPointer { dstPtr in
            dstPtr.baseAddress!.update(
              from: srcPtr.baseAddress! + readIndex, count: first)
            dstPtr.baseAddress!.advanced(by: first).update(
              from: srcPtr.baseAddress!, count: takeCount - first)
          }
        }
      }
      return takeCount
    }

    var isEmpty: Bool { count == 0 }

    mutating func removeAll(keepingCapacity: Bool = false) {
      readIndex = 0
      writeIndex = 0
      count = 0
    }
  }

  // Ring buffers with 4096 sample capacity (~256ms at 16kHz).
  // Mutated directly through the stored properties — going through a
  // computed-property wrapper here would force a CoW copy of the 16KB
  // storage on every append (the get/set returns/restores a struct copy
  // whose [Float] storage is shared via CoW until the first mutating
  // write). Audio callbacks happen frequently enough that cost is real.
  private static let ringBufferCapacity = 4096
  private var pendingMicRing: RingBuffer
  private var pendingLoopbackRing: RingBuffer

  // MARK: - Public API

  /// Feed far-end (loopback/system audio) samples.
  /// Call this BEFORE processNearEnd for proper echo reference.
  /// - Parameter samples: Audio samples at 16kHz, Float format
  public func feedFarEnd(_ samples: [Float]) {
    os_unfair_lock_lock(&processingLock)
    defer { os_unfair_lock_unlock(&processingLock) }
    pendingLoopbackRing.append(contentsOf: samples)
  }

  /// Process near-end (microphone) samples and return echo-cancelled output.
  /// Returns the same number of samples as input (with processing delay).
  /// - Parameter samples: Microphone audio at 16kHz, Float format
  /// - Returns: Echo-cancelled audio samples
  public func processNearEnd(_ samples: [Float]) -> [Float] {
    os_unfair_lock_lock(&processingLock)
    defer { os_unfair_lock_unlock(&processingLock) }

    // isInitialized must be checked under the lock — concurrent unloadModels
    // could nil the model pointers between guard and use otherwise.
    guard isInitialized else {
      logger.warning("DTLN-aec not initialized, passing through")
      return samples
    }

    pendingMicRing.append(contentsOf: samples)
    // Pre-size the output: every full frame produces exactly blockShift
    // output samples. Avoids ~log2(N) reallocations as outputSamples grows.
    let pendingCount = min(pendingMicRing.count, pendingLoopbackRing.count)
    let expectedFrames = pendingCount / Self.blockShift
    var outputSamples: [Float] = []
    outputSamples.reserveCapacity(expectedFrames * Self.blockShift)

    // Process in blockShift (128 sample) chunks - Python style
    while pendingMicRing.count >= Self.blockShift
      && pendingLoopbackRing.count >= Self.blockShift
    {
      let frameStart = Date()

      // Python-style sliding window: shift left and add new samples at end.
      // micBuffer and loopbackBuffer are always exactly blockLen (512) samples.
      // Read directly into the preallocated prefixReadBuffer to avoid the
      // [Float] allocation that RingBuffer.prefix(_:) makes per call.
      pendingMicRing.copyPrefix(into: &prefixReadBuffer, count: Self.blockShift)
      let newMicPeak = shiftAndAppend(buffer: &micBuffer, newSamples: prefixReadBuffer)
      pendingLoopbackRing.copyPrefix(into: &prefixReadBuffer, count: Self.blockShift)
      let newLpbPeak = shiftAndAppend(buffer: &loopbackBuffer, newSamples: prefixReadBuffer)
      // Rolling peak estimate for the 512-sample window: take the new 128
      // samples' peak and decay the previous estimate (0.75 ≈ 19ms half-life
      // at 8ms/frame), avoiding a full vDSP_maxmgv scan per frame. This
      // feeds LinkedGainController's own 1ms-attack/100ms-release envelope —
      // the two stages are intentionally layered (cheap rolling peak +
      // perceptual envelope), not redundant.
      micBufferPeak = max(micBufferPeak * 0.75, newMicPeak)
      loopbackBufferPeak = max(loopbackBufferPeak * 0.75, newLpbPeak)

      pendingMicRing.removeFirst(Self.blockShift)
      pendingLoopbackRing.removeFirst(Self.blockShift)

      // Process the current 512-sample window
      if let processed = processFrame(mic: micBuffer, loopback: loopbackBuffer) {
        overlapAddAppend(processed, to: &outputSamples)
      } else {
        overlapAddAppend(micBuffer, to: &outputSamples)
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
    return pendingMicRing.count + overlapTail
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
    os_unfair_lock_lock(&processingLock)
    defer { os_unfair_lock_unlock(&processingLock) }

    guard isInitialized else {
      let samples = pendingMicRing.prefix(pendingMicRing.count)
      pendingMicRing.removeAll(keepingCapacity: true)
      pendingLoopbackRing.removeAll(keepingCapacity: true)
      return samples
    }

    var outputSamples: [Float] = []

    // Save gain controller state before processing zero-padded frames
    // to preserve session continuity (matching LSTM state preservation)
    let savedGainState = gainController?.captureState()

    // Process any pending samples by zero-padding to blockShift boundary
    if !pendingMicRing.isEmpty || !pendingLoopbackRing.isEmpty {
      // Pad mic samples to blockShift
      let micPadCount = Self.blockShift - pendingMicRing.count
      if micPadCount > 0 {
        pendingMicRing.append(contentsOf: [Float](repeating: 0, count: micPadCount))
      }

      // Pad loopback samples to blockShift
      let lpbPadCount = Self.blockShift - pendingLoopbackRing.count
      if lpbPadCount > 0 {
        pendingLoopbackRing.append(contentsOf: [Float](repeating: 0, count: lpbPadCount))
      }

      // Process the padded frame through normal path
      pendingMicRing.copyPrefix(into: &prefixReadBuffer, count: Self.blockShift)
      let newMicPeak = shiftAndAppend(buffer: &micBuffer, newSamples: prefixReadBuffer)
      pendingLoopbackRing.copyPrefix(into: &prefixReadBuffer, count: Self.blockShift)
      let newLpbPeak = shiftAndAppend(buffer: &loopbackBuffer, newSamples: prefixReadBuffer)
      micBufferPeak = max(micBufferPeak * 0.75, newMicPeak)
      loopbackBufferPeak = max(loopbackBufferPeak * 0.75, newLpbPeak)

      pendingMicRing.removeAll(keepingCapacity: true)
      pendingLoopbackRing.removeAll(keepingCapacity: true)

      if let processed = processFrame(mic: micBuffer, loopback: loopbackBuffer) {
        overlapAddAppend(processed, to: &outputSamples)
      } else {
        overlapAddAppend(micBuffer, to: &outputSamples)
      }
    }

    // Restore gain controller state to preserve session continuity
    if let state = savedGainState {
      gainController?.restoreState(state)
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
  /// - Returns: Peak absolute value of the new samples being appended
  @discardableResult
  private func shiftAndAppend(buffer: inout [Float], newSamples: [Float]) -> Float {
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

    // Compute peak of new samples (128 samples vs full 512 buffer scan)
    var peak: Float = 0
    if !newSamples.isEmpty {
      vDSP_maxmgv(newSamples, 1, &peak, vDSP_Length(newSamples.count))
    }
    return peak
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

    // Apply linked gain control if enabled
    var effectiveMic: [Float]
    var effectiveLoopback: [Float]

    if let gc = gainController {
      // Use tracked peaks from shiftAndAppend (avoids full 512-sample buffer scan)
      let linkedGain = gc.computeLinkedGain(micPeak: micBufferPeak, sysPeak: loopbackBufferPeak)

      // Apply same gain to both streams using pre-allocated buffers (zero allocation)
      vDSP.multiply(linkedGain, mic, result: &effectiveMicBuffer)
      vDSP.multiply(linkedGain, loopback, result: &effectiveLoopbackBuffer)
      effectiveMic = effectiveMicBuffer
      effectiveLoopback = effectiveLoopbackBuffer
    } else {
      effectiveMic = mic
      effectiveLoopback = loopback
    }

    // Part 1: Frequency Domain
    // Mic FFT goes into the persistent mic split-complex so applyMaskAndIFFT
    // can reuse it; loopback FFT lands in the scratch buffer (magnitude only).
    // Magnitudes write directly into the per-frame magnitude ivars.
    forwardFFT(samples: effectiveMic, realBuf: &micFftRealBuffer, imagBuf: &micFftImagBuffer)
    magnitudeFromFFT(
      realBuf: &micFftRealBuffer, imagBuf: &micFftImagBuffer, magOut: &micMagBuffer)

    forwardFFT(samples: effectiveLoopback, realBuf: &fftRealBuffer, imagBuf: &fftImagBuffer)
    magnitudeFromFFT(
      realBuf: &fftRealBuffer, imagBuf: &fftImagBuffer, magOut: &lpbMagBuffer)

    copyToMLArray(micMagBuffer, to: micMagArray)
    copyToMLArray(lpbMagBuffer, to: lpbMagArray)

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
    applyMaskAndIFFT(
      realBuf: &micFftRealBuffer, imagBuf: &micFftImagBuffer, mask: mask,
      output: &frameOutputBuffer)

    // Part 2: Time Domain
    copyToMLArray(frameOutputBuffer, to: estimatedFrameArray)
    copyToMLArray(effectiveLoopback, to: lpbTimeArray)

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
    extractFromMLArrayInto(output, &frameResultBuffer, count: Self.blockLen)

    // Validate numerics if enabled
    if config.validateNumerics && containsInvalidNumerics(frameResultBuffer) {
      // Fall back to passthrough (gain-adjusted mic input) on NaN/Inf
      return effectiveMic
    }

    // Clip output if enabled
    if config.clipOutput {
      clipToValidRange(&frameResultBuffer)
    }

    return frameResultBuffer
  }

  // MARK: - FFT Helpers

  /// Forward real FFT of `samples` into the supplied split-complex buffers
  /// (vDSP packed format: DC in `realBuf[0]`, Nyquist in `imagBuf[0]`).
  /// Buffers must each be `blockLen / 2` floats.
  private func forwardFFT(
    samples: [Float], realBuf: inout [Float], imagBuf: inout [Float]
  ) {
    guard let fftSetup else { return }
    let halfLen = Self.blockLen / 2
    samples.withUnsafeBufferPointer { srcPtr in
      realBuf.withUnsafeMutableBufferPointer { realPtr in
        imagBuf.withUnsafeMutableBufferPointer { imagPtr in
          var splitComplex = DSPSplitComplex(
            realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!)
          // De-interleave samples into split-complex (even→real, odd→imag) —
          // the standard preprocessing for vDSP's packed-real forward FFT.
          srcPtr.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: halfLen) {
            cPtr in
            vDSP_ctoz(cPtr, 2, &splitComplex, 1, vDSP_Length(halfLen))
          }
          vDSP_fft_zrip(fftSetup, &splitComplex, 1, Self.fftLog2n, FFTDirection(FFT_FORWARD))
        }
      }
    }
  }

  /// Compute magnitude spectrum (length `fftBins`) from a packed-real
  /// split-complex previously written by `forwardFFT`, writing into the
  /// provided `magOut` buffer (must already be sized to `fftBins`).
  /// Output is scaled by 0.5 to compensate vDSP's forward-FFT scale-by-2
  /// (matches NumPy rfft). Real/imag buffers are taken `inout` because
  /// `DSPSplitComplex` requires mutable pointers; their contents are not
  /// modified.
  private func magnitudeFromFFT(
    realBuf: inout [Float], imagBuf: inout [Float], magOut: inout [Float]
  ) {
    realBuf.withUnsafeMutableBufferPointer { realPtr in
      imagBuf.withUnsafeMutableBufferPointer { imagPtr in
        // DC (purely real) and Nyquist (purely real) live in realp[0] / imagp[0].
        magOut[0] = abs(realPtr[0])
        magOut[Self.fftBins - 1] = abs(imagPtr[0])

        // Bins 1..255: |z|^2 via vDSP, then sqrt.
        var midSplit = DSPSplitComplex(
          realp: realPtr.baseAddress! + 1, imagp: imagPtr.baseAddress! + 1)
        magOut.withUnsafeMutableBufferPointer { magPtr in
          vDSP_zvmags(&midSplit, 1, magPtr.baseAddress! + 1, 1, vDSP_Length(Self.fftBins - 2))
        }
      }
    }

    // Clamp to zero before sqrt to prevent NaN from floating-point errors.
    var lowerBound: Float = 0.0
    var upperBound: Float = .greatestFiniteMagnitude
    magOut.withUnsafeMutableBufferPointer { magPtr in
      vDSP_vclip(
        magPtr.baseAddress! + 1, 1, &lowerBound, &upperBound,
        magPtr.baseAddress! + 1, 1, vDSP_Length(Self.fftBins - 2))
    }

    var sqrtCount = Int32(Self.fftBins - 2)
    magOut.withUnsafeMutableBufferPointer { magPtr in
      vvsqrtf(magPtr.baseAddress! + 1, magPtr.baseAddress! + 1, &sqrtCount)
    }

    // Compensate vDSP forward-FFT scale-by-2 to match NumPy rfft amplitudes.
    vDSP.multiply(0.5, magOut, result: &magOut)
  }

  /// Apply the spectral `mask` to a previously-computed mic split-complex
  /// (in `realBuf` / `imagBuf`), inverse FFT, and write the time-domain
  /// frame of length `blockLen` into `output` (must be sized to `blockLen`).
  ///
  /// The forward FFT must already live in the supplied buffers — this
  /// function does NOT recompute it. Saves one 512-pt forward FFT per
  /// frame compared to the previous "re-FFT from samples" approach.
  private func applyMaskAndIFFT(
    realBuf: inout [Float], imagBuf: inout [Float], mask: MLMultiArray,
    output: inout [Float]
  ) {
    guard let fftSetup else {
      // Zero out — caller will see silence but won't NaN.
      for i in 0..<output.count { output[i] = 0 }
      return
    }

    let halfLen = Self.blockLen / 2

    // Pre-extract mask into fp32 scratch so the per-bin loop can use vDSP_vmul
    // and avoid a fp16/fp32 branch per bin. One vImage call handles both
    // precisions; for fp32 input it's a memcpy.
    extractFromMLArrayInto(mask, &maskScratch, count: Self.fftBins)

    realBuf.withUnsafeMutableBufferPointer { realPtr in
      imagBuf.withUnsafeMutableBufferPointer { imagPtr in
        maskScratch.withUnsafeBufferPointer { maskPtr in
          var splitComplex = DSPSplitComplex(
            realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!)

          // Packed DC/Nyquist live in realp[0]/imagp[0]; multiply scalarly.
          realPtr[0] *= maskPtr[0]
          imagPtr[0] *= maskPtr[Self.fftBins - 1]

          // Regular bins 1..fftBins-2: in-place vector multiply on real + imag.
          let midCount = vDSP_Length(Self.fftBins - 2)
          vDSP_vmul(
            realPtr.baseAddress! + 1, 1, maskPtr.baseAddress! + 1, 1,
            realPtr.baseAddress! + 1, 1, midCount)
          vDSP_vmul(
            imagPtr.baseAddress! + 1, 1, maskPtr.baseAddress! + 1, 1,
            imagPtr.baseAddress! + 1, 1, midCount)

          // Inverse real FFT - vDSP handles conjugate symmetry internally.
          vDSP_fft_zrip(fftSetup, &splitComplex, 1, Self.fftLog2n, FFTDirection(FFT_INVERSE))

          // Unpack split-complex back to interleaved real samples.
          output.withUnsafeMutableBufferPointer { outPtr in
            outPtr.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: halfLen) {
              cPtr in
              vDSP_ztoc(&splitComplex, 1, cPtr, 2, vDSP_Length(halfLen))
            }
          }
        }
      }
    }

    // vDSP real FFT scales by 2 on forward and 2 on inverse = 4x total.
    // Divide by 2*N to get correct amplitude.
    vDSP.multiply(1.0 / Float(2 * Self.blockLen), output, result: &output)
  }

  // MARK: - Overlap-Add

  /// Add `frame` into `outputBuffer`, append the leading `blockShift`
  /// samples directly to `outputSamples`, and shift `outputBuffer` left.
  /// Avoids the fresh `[Float]` allocation that returning a prefix array
  /// would produce per frame.
  private func overlapAddAppend(_ frame: [Float], to outputSamples: inout [Float]) {
    vDSP.add(outputBuffer, frame, result: &outputBuffer)
    // ArraySlice; `append(contentsOf:)` reads it via memcpy without allocation.
    outputSamples.append(contentsOf: outputBuffer.prefix(Self.blockShift))

    let overlapCount = Self.blockLen - Self.blockShift  // 384
    // Source [128, 512) and dest [0, 384) overlap on [128, 384) — must be memmove,
    // not memcpy. UnsafeMutablePointer.update(from:count:) is memcpy semantics.
    _ = outputBuffer.withUnsafeMutableBytes { rawPtr in
      memmove(
        rawPtr.baseAddress!,
        rawPtr.baseAddress! + Self.blockShift * MemoryLayout<Float>.stride,
        overlapCount * MemoryLayout<Float>.stride)
    }
    outputBuffer.withUnsafeMutableBufferPointer { ptr in
      ptr.baseAddress!.advanced(by: overlapCount).update(repeating: 0, count: Self.blockShift)
    }
  }

  // MARK: - MLMultiArray Helpers

  /// Copy `source` (fp32) into `array`. fp32 → fp32 uses `memcpy`; fp32 →
  /// fp16 uses vImage's planar conversion. Both replace what was a scalar
  /// per-element loop on the per-frame path.
  private func copyToMLArray(_ source: [Float], to array: MLMultiArray) {
    let n = min(source.count, array.count)
    source.withUnsafeBufferPointer { srcPtr in
      if array.dataType == .float16 {
        var src = vImage_Buffer(
          data: UnsafeMutableRawPointer(mutating: srcPtr.baseAddress!),
          height: 1, width: vImagePixelCount(n),
          rowBytes: n * MemoryLayout<Float>.stride)
        var dst = vImage_Buffer(
          data: array.dataPointer,
          height: 1, width: vImagePixelCount(n),
          rowBytes: n * MemoryLayout<Float16>.stride)
        vImageConvert_PlanarFtoPlanar16F(&src, &dst, vImage_Flags(kvImageNoFlags))
      } else {
        memcpy(array.dataPointer, srcPtr.baseAddress!, n * MemoryLayout<Float>.stride)
      }
    }
  }

  /// Extract `count` floats from `array` into the provided `dest` buffer
  /// (must already be sized to `count`). fp32 source uses `memcpy`; fp16
  /// source uses vImage's planar 16F→F conversion.
  private func extractFromMLArrayInto(
    _ array: MLMultiArray, _ dest: inout [Float], count: Int
  ) {
    let n = min(count, array.count)
    dest.withUnsafeMutableBufferPointer { dstPtr in
      if array.dataType == .float16 {
        var src = vImage_Buffer(
          data: array.dataPointer,
          height: 1, width: vImagePixelCount(n),
          rowBytes: n * MemoryLayout<Float16>.stride)
        var dst = vImage_Buffer(
          data: dstPtr.baseAddress!,
          height: 1, width: vImagePixelCount(n),
          rowBytes: n * MemoryLayout<Float>.stride)
        vImageConvert_Planar16FtoPlanarF(&src, &dst, vImage_Flags(kvImageNoFlags))
      } else {
        memcpy(dstPtr.baseAddress!, array.dataPointer, n * MemoryLayout<Float>.stride)
      }
    }
  }

  /// Copy LSTM-state contents from `source` to `dest`. Same-precision
  /// pairs go through `memcpy`; mixed-precision pairs use vImage's planar
  /// conversion. Replaces a scalar per-element loop on a 1024-element
  /// buffer that ran twice per frame.
  private func copyStates(from source: MLMultiArray, to dest: MLMultiArray) {
    let n = min(source.count, dest.count)
    let srcIs16 = source.dataType == .float16
    let dstIs16 = dest.dataType == .float16

    switch (srcIs16, dstIs16) {
    case (false, false):
      memcpy(dest.dataPointer, source.dataPointer, n * MemoryLayout<Float>.stride)
    case (true, true):
      memcpy(dest.dataPointer, source.dataPointer, n * MemoryLayout<Float16>.stride)
    case (true, false):
      var src = vImage_Buffer(
        data: source.dataPointer, height: 1, width: vImagePixelCount(n),
        rowBytes: n * MemoryLayout<Float16>.stride)
      var dst = vImage_Buffer(
        data: dest.dataPointer, height: 1, width: vImagePixelCount(n),
        rowBytes: n * MemoryLayout<Float>.stride)
      vImageConvert_Planar16FtoPlanarF(&src, &dst, vImage_Flags(kvImageNoFlags))
    case (false, true):
      var src = vImage_Buffer(
        data: source.dataPointer, height: 1, width: vImagePixelCount(n),
        rowBytes: n * MemoryLayout<Float>.stride)
      var dst = vImage_Buffer(
        data: dest.dataPointer, height: 1, width: vImagePixelCount(n),
        rowBytes: n * MemoryLayout<Float16>.stride)
      vImageConvert_PlanarFtoPlanar16F(&src, &dst, vImage_Flags(kvImageNoFlags))
    }
  }

  // MARK: - Numeric Validation

  /// Check if array contains any NaN or Inf values.
  /// Uses sum-of-squares as a single vectorized reduction: NaN propagates
  /// (NaN² = NaN, Σ NaN = NaN), and ±Inf squared is +Inf which also fails
  /// `isFinite`. Replaces a 512-iter scalar branch loop on the hot path.
  private func containsInvalidNumerics(_ array: [Float]) -> Bool {
    var sumOfSquares: Float = 0
    array.withUnsafeBufferPointer { ptr in
      vDSP_svesq(ptr.baseAddress!, 1, &sumOfSquares, vDSP_Length(array.count))
    }
    return !sumOfSquares.isFinite
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
