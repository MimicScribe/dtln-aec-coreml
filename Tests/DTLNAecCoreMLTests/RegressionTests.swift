// RegressionTests.swift
// DTLN-aec CoreML Regression Tests
//
// IMPORTANT: These tests verify that model output quality meets or exceeds
// established baselines. A passing test means NO REGRESSION from baseline values.
//
// Each test compares current output metrics (dB reduction, RMS, etc.) against
// the baselines defined in Tests/Baselines/regression_baselines.json. Tests fail
// if metrics fall below baseline thresholds (with tolerance), indicating a
// regression in echo cancellation quality.
//
// Baseline values should be updated when intentional improvements are made.

import Foundation
import XCTest

@testable import DTLNAecCoreML

// MARK: - Baseline Data Structures

struct RegressionBaselines: Codable {
  let version: String
  let date: String
  let description: String
  let tolerance: Tolerance
  let models: [String: ModelBaseline]

  struct Tolerance: Codable {
    let db: Float
    let rms_ratio: Float
    let convergence_ratio: Float?
    let spectral_pct: Float?
  }

  struct ModelBaseline: Codable {
    let aec_challenge: AECChallengeBaseline?
    let realworld: RealworldBaseline?
    let python_reference: PythonReferenceBaseline?
  }

  struct SpectralEnergy: Codable {
    let low_pct: Float
    let mid_pct: Float
    let high_pct: Float
  }

  struct SegmentedRMS: Codable {
    let min: Float
    let max: Float
    let mean: Float
    let stddev: Float
  }

  struct TimeEvolution: Codable {
    // Using string keys for flexibility with time points
    let values: [String: Float]?

    init(from decoder: Decoder) throws {
      let container = try decoder.singleValueContainer()
      values = try? container.decode([String: Float].self)
    }

    func encode(to encoder: Encoder) throws {
      var container = encoder.singleValueContainer()
      try container.encode(values)
    }
  }

  struct AECChallengeBaseline: Codable {
    let reduction_db: Float
    let output_rms: Float
    let peak_amplitude: Float?
    let crest_factor_db: Float?
    let convergence_ratio: Float?
    let spectral_energy: SpectralEnergy?
    let segmented_rms: SegmentedRMS?
    let description: String
  }

  struct RealworldBaseline: Codable {
    let reduction_db: Float
    let output_rms: Float
    let peak_amplitude: Float?
    let crest_factor_db: Float?
    let convergence_ratio: Float?
    let spectral_energy: SpectralEnergy?
    let segmented_rms: SegmentedRMS?
    let time_evolution: [String: Float]?
    let description: String
  }

  struct PythonReferenceBaseline: Codable {
    let python_rms: Float
    let coreml_rms: Float
    let rms_ratio: Float
    let max_acceptable_rms: Float
    let max_rms_ratio: Float
    let description: String
  }
}

// MARK: - Shared Test Utilities

/// Shared utilities for regression tests
enum RegressionTestUtils {

  // MARK: - Baseline Loading

  private static var _cachedBaselines: RegressionBaselines?

  static func loadBaselines() throws -> RegressionBaselines {
    if let cached = _cachedBaselines {
      return cached
    }

    let baselinesURL = packageRoot().appendingPathComponent(
      "Tests/Baselines/regression_baselines.json")
    let data = try Data(contentsOf: baselinesURL)
    let baselines = try JSONDecoder().decode(RegressionBaselines.self, from: data)
    _cachedBaselines = baselines
    return baselines
  }

  static func baseline(for model: String) throws -> RegressionBaselines.ModelBaseline {
    let baselines = try loadBaselines()
    guard let modelBaseline = baselines.models[model] else {
      throw NSError(
        domain: "Baselines", code: 1,
        userInfo: [NSLocalizedDescriptionKey: "No baseline found for model: \(model)"])
    }
    return modelBaseline
  }

  static func tolerance() throws -> RegressionBaselines.Tolerance {
    return try loadBaselines().tolerance
  }

  // MARK: - WAV File Reading

  /// Read WAV file, supporting both Int16 and Float32 formats
  static func readWAVFile(_ url: URL) throws -> [Float] {
    let data = try Data(contentsOf: url)
    guard data.count > 44 else {
      throw NSError(
        domain: "WAV", code: 1, userInfo: [NSLocalizedDescriptionKey: "File too small"])
    }

    // Parse format chunk to determine audio format
    var audioFormat: UInt16 = 0
    var bitsPerSample: UInt16 = 0
    var dataOffset = 0
    var dataSize = 0

    var offset = 12  // Skip RIFF header
    while offset < data.count - 8 {
      let chunkID = String(bytes: data[offset..<offset + 4], encoding: .ascii) ?? ""
      let chunkSize = data.withUnsafeBytes { ptr in
        ptr.load(fromByteOffset: offset + 4, as: UInt32.self)
      }

      if chunkID == "fmt " {
        audioFormat = data.withUnsafeBytes { ptr in
          ptr.load(fromByteOffset: offset + 8, as: UInt16.self)
        }
        bitsPerSample = data.withUnsafeBytes { ptr in
          ptr.load(fromByteOffset: offset + 22, as: UInt16.self)
        }
      } else if chunkID == "data" {
        dataOffset = offset + 8
        dataSize = Int(chunkSize)
        break
      }
      offset += 8 + Int(chunkSize)
    }

    guard dataOffset > 0 else {
      throw NSError(
        domain: "WAV", code: 2, userInfo: [NSLocalizedDescriptionKey: "No data chunk found"])
    }

    // Read samples based on format
    if audioFormat == 1 && bitsPerSample == 16 {
      // PCM Int16
      let sampleCount = dataSize / 2
      var samples = [Float](repeating: 0, count: sampleCount)
      data.withUnsafeBytes { ptr in
        for i in 0..<sampleCount {
          let int16Value = ptr.load(fromByteOffset: dataOffset + i * 2, as: Int16.self)
          samples[i] = Float(int16Value) / 32768.0
        }
      }
      return samples
    } else if audioFormat == 3 && bitsPerSample == 32 {
      // IEEE Float32
      let sampleCount = dataSize / 4
      var samples = [Float](repeating: 0, count: sampleCount)
      data.withUnsafeBytes { ptr in
        for i in 0..<sampleCount {
          samples[i] = ptr.load(fromByteOffset: dataOffset + i * 4, as: Float.self)
        }
      }
      return samples
    } else {
      throw NSError(
        domain: "WAV", code: 3,
        userInfo: [
          NSLocalizedDescriptionKey: "Unsupported format: \(audioFormat), \(bitsPerSample)-bit"
        ])
    }
  }

  // MARK: - Helper Functions

  static func computeCorrelation(_ a: [Float], _ b: [Float]) -> Float {
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

  static func computeRMS(_ samples: [Float]) -> Float {
    guard !samples.isEmpty else { return 0 }
    let sumSquares = samples.reduce(0) { $0 + $1 * $1 }
    return sqrt(sumSquares / Float(samples.count))
  }

  static func computeRMSError(_ a: [Float], _ b: [Float]) -> Float {
    let length = min(a.count, b.count)
    guard length > 0 else { return 0 }

    var sumSquaredError: Float = 0
    for i in 0..<length {
      let diff = a[i] - b[i]
      sumSquaredError += diff * diff
    }
    return sqrt(sumSquaredError / Float(length))
  }

  static func computeReductionDB(_ inputRMS: Float, _ outputRMS: Float) -> Float {
    guard outputRMS > 0 else { return .infinity }
    return 20 * log10(inputRMS / outputRMS)
  }

  // MARK: - Advanced Diagnostic Metrics

  /// Peak absolute amplitude
  static func computePeakAmplitude(_ samples: [Float]) -> Float {
    return samples.map { abs($0) }.max() ?? 0
  }

  /// Crest factor (peak-to-RMS ratio) - indicates spikiness
  /// Higher values = more transient/spiky, lower = more consistent
  static func computeCrestFactor(_ samples: [Float]) -> Float {
    let rms = computeRMS(samples)
    let peak = computePeakAmplitude(samples)
    guard rms > 0 else { return 0 }
    return peak / rms
  }

  /// Crest factor in dB
  static func computeCrestFactorDB(_ samples: [Float]) -> Float {
    let crestFactor = computeCrestFactor(samples)
    guard crestFactor > 0 else { return 0 }
    return 20 * log10(crestFactor)
  }

  /// Time-segmented RMS analysis
  /// Returns (min, max, mean, stddev) of RMS values across segments
  static func computeSegmentedRMS(
    _ samples: [Float], segmentMs: Int = 100, sampleRate: Int = 16000
  )
    -> (min: Float, max: Float, mean: Float, stddev: Float)
  {
    let segmentSize = (sampleRate * segmentMs) / 1000
    guard segmentSize > 0, samples.count >= segmentSize else {
      let rms = computeRMS(samples)
      return (rms, rms, rms, 0)
    }

    var segmentRMSValues: [Float] = []
    var offset = 0
    while offset + segmentSize <= samples.count {
      let segment = Array(samples[offset..<(offset + segmentSize)])
      segmentRMSValues.append(computeRMS(segment))
      offset += segmentSize
    }

    guard !segmentRMSValues.isEmpty else {
      return (0, 0, 0, 0)
    }

    let minRMS = segmentRMSValues.min() ?? 0
    let maxRMS = segmentRMSValues.max() ?? 0
    let meanRMS = segmentRMSValues.reduce(0, +) / Float(segmentRMSValues.count)

    // Standard deviation
    let variance =
      segmentRMSValues.reduce(0) { $0 + ($1 - meanRMS) * ($1 - meanRMS) }
      / Float(segmentRMSValues.count)
    let stddev = sqrt(variance)

    return (minRMS, maxRMS, meanRMS, stddev)
  }

  /// Simple spectral energy analysis using band-pass approximation
  /// Returns energy in low (0-500Hz), mid (500-2000Hz), and high (2000-8000Hz) bands
  /// Uses a simple DFT approach for analysis (not optimized, but sufficient for testing)
  static func computeSpectralEnergy(
    _ samples: [Float], sampleRate: Int = 16000
  )
    -> (low: Float, mid: Float, high: Float)
  {
    // Use 512-sample FFT frames, hop by 256
    let fftSize = 512
    let hopSize = 256
    guard samples.count >= fftSize else {
      return (0, 0, 0)
    }

    // Frequency bin width
    let binWidth = Float(sampleRate) / Float(fftSize)

    // Band boundaries in bins
    let lowEnd = Int(500.0 / binWidth)  // 0-500 Hz
    let midEnd = Int(2000.0 / binWidth)  // 500-2000 Hz
    let highEnd = Int(8000.0 / binWidth)  // 2000-8000 Hz (or Nyquist)

    var lowEnergy: Float = 0
    var midEnergy: Float = 0
    var highEnergy: Float = 0
    var frameCount = 0

    var offset = 0
    while offset + fftSize <= samples.count {
      let frame = Array(samples[offset..<(offset + fftSize)])

      // Apply Hann window
      var windowed = [Float](repeating: 0, count: fftSize)
      for i in 0..<fftSize {
        let window = 0.5 * (1 - cos(2 * Float.pi * Float(i) / Float(fftSize - 1)))
        windowed[i] = frame[i] * window
      }

      // Compute magnitude spectrum using DFT (just positive frequencies)
      for k in 0..<(fftSize / 2) {
        var real: Float = 0
        var imag: Float = 0
        for n in 0..<fftSize {
          let angle = -2 * Float.pi * Float(k) * Float(n) / Float(fftSize)
          real += windowed[n] * cos(angle)
          imag += windowed[n] * sin(angle)
        }
        let magnitude = sqrt(real * real + imag * imag)
        let energy = magnitude * magnitude

        if k < lowEnd {
          lowEnergy += energy
        } else if k < midEnd {
          midEnergy += energy
        } else if k < highEnd {
          highEnergy += energy
        }
      }

      frameCount += 1
      offset += hopSize
    }

    // Average across frames
    guard frameCount > 0 else { return (0, 0, 0) }
    return (
      lowEnergy / Float(frameCount),
      midEnergy / Float(frameCount),
      highEnergy / Float(frameCount)
    )
  }

  /// Convert energy to dB relative to reference
  static func energyToDBFS(_ energy: Float, reference: Float = 1.0) -> Float {
    guard energy > 0 else { return -Float.infinity }
    return 10 * log10(energy / reference)
  }

  /// Time evolution analysis - shows how RMS changes over the recording
  /// Divides signal into N equal chunks and returns RMS for each
  static func computeTimeEvolution(
    _ samples: [Float], chunks: Int = 10, sampleRate: Int = 16000
  )
    -> [(timeSeconds: Float, rms: Float)]
  {
    guard chunks > 0, !samples.isEmpty else { return [] }

    let chunkSize = samples.count / chunks
    guard chunkSize > 0 else { return [] }

    var results: [(timeSeconds: Float, rms: Float)] = []

    for i in 0..<chunks {
      let start = i * chunkSize
      let end = min(start + chunkSize, samples.count)
      let chunk = Array(samples[start..<end])
      let rms = computeRMS(chunk)
      let timeSeconds = Float(start + chunkSize / 2) / Float(sampleRate)
      results.append((timeSeconds, rms))
    }

    return results
  }

  /// Compute convergence metric - ratio of first half RMS to second half RMS
  /// Values > 1 mean signal starts loud and gets quieter (slow convergence)
  /// Values < 1 mean signal starts quiet and gets louder
  /// Values ≈ 1 mean consistent processing
  static func computeConvergenceRatio(_ samples: [Float]) -> Float {
    guard samples.count >= 2 else { return 1.0 }

    let midpoint = samples.count / 2
    let firstHalf = Array(samples[0..<midpoint])
    let secondHalf = Array(samples[midpoint..<samples.count])

    let firstRMS = computeRMS(firstHalf)
    let secondRMS = computeRMS(secondHalf)

    guard secondRMS > 0 else { return Float.infinity }
    return firstRMS / secondRMS
  }

  /// Comprehensive audio analysis report
  struct AudioAnalysis {
    let rms: Float
    let peakAmplitude: Float
    let crestFactorDB: Float
    let segmentedRMS: (min: Float, max: Float, mean: Float, stddev: Float)
    let spectralEnergy: (low: Float, mid: Float, high: Float)
    let convergenceRatio: Float
    let timeEvolution: [(timeSeconds: Float, rms: Float)]

    func printReport(label: String) {
      print("\n  --- \(label) Analysis ---")
      print("  RMS: \(String(format: "%.6f", rms))")
      print("  Peak amplitude: \(String(format: "%.6f", peakAmplitude))")
      print("  Crest factor: \(String(format: "%.1f", crestFactorDB)) dB")
      print(
        "  Convergence ratio (1st half / 2nd half): \(String(format: "%.2f", convergenceRatio))")
      if convergenceRatio > 1.2 {
        print("    ⚠️  Signal starts loud, gets quieter (slow state convergence)")
      } else if convergenceRatio < 0.8 {
        print("    Signal starts quiet, gets louder")
      } else {
        print("    ✓ Consistent processing throughout")
      }
      print("  Time evolution (RMS over time):")
      for point in timeEvolution {
        let bar = String(repeating: "█", count: min(50, Int(point.rms * 500)))
        print(
          "    \(String(format: "%5.1f", point.timeSeconds))s: \(String(format: "%.6f", point.rms)) \(bar)"
        )
      }
      print("  Segmented RMS (100ms):")
      print("    Min: \(String(format: "%.6f", segmentedRMS.min))")
      print("    Max: \(String(format: "%.6f", segmentedRMS.max))")
      print("    Mean: \(String(format: "%.6f", segmentedRMS.mean))")
      print("    StdDev: \(String(format: "%.6f", segmentedRMS.stddev))")
      print("  Spectral energy:")
      let totalSpectral = spectralEnergy.low + spectralEnergy.mid + spectralEnergy.high
      let lowPct = totalSpectral > 0 ? 100 * spectralEnergy.low / totalSpectral : 0
      let midPct = totalSpectral > 0 ? 100 * spectralEnergy.mid / totalSpectral : 0
      let highPct = totalSpectral > 0 ? 100 * spectralEnergy.high / totalSpectral : 0
      print(
        "    Low (0-500Hz): \(String(format: "%.1f", lowPct))%"
      )
      print(
        "    Mid (500-2kHz): \(String(format: "%.1f", midPct))%"
      )
      print(
        "    High (2k-8kHz): \(String(format: "%.1f", highPct))%"
      )
    }
  }

  /// Perform comprehensive analysis on audio samples
  static func analyzeAudio(_ samples: [Float]) -> AudioAnalysis {
    return AudioAnalysis(
      rms: computeRMS(samples),
      peakAmplitude: computePeakAmplitude(samples),
      crestFactorDB: computeCrestFactorDB(samples),
      segmentedRMS: computeSegmentedRMS(samples),
      spectralEnergy: computeSpectralEnergy(samples),
      convergenceRatio: computeConvergenceRatio(samples),
      timeEvolution: computeTimeEvolution(samples)
    )
  }

  /// Get package root from test file location
  static func packageRoot(from file: String = #file) -> URL {
    URL(fileURLWithPath: file)
      .deletingLastPathComponent()  // DTLNAecCoreMLTests/
      .deletingLastPathComponent()  // Tests/
      .deletingLastPathComponent()  // package root
  }

  /// Get samples directory
  static func samplesDir(from file: String = #file) -> URL {
    packageRoot(from: file).appendingPathComponent("Samples/aec_challenge")
  }
}

// MARK: - 128-Unit Model Regression Tests

/// Regression tests for the 128-unit (small) model
final class RegressionTests128: XCTestCase {

  /// Strict regression test: CoreML output should match Python reference
  /// The original Python DTLN-aec achieves near-silence on this echo-only sample
  func testCoreMLMatchesPythonReference() throws {
    let baseline = try RegressionTestUtils.baseline(for: "128-unit")
    guard let pythonRef = baseline.python_reference else {
      throw XCTSkip("No Python reference baseline defined for 128-unit model")
    }
    let tolerance = try RegressionTestUtils.tolerance()

    let samplesDir = RegressionTestUtils.samplesDir()
    let pythonFile = samplesDir.appendingPathComponent("farend_singletalk_processed_python_128.wav")
    let coremlFile = samplesDir.appendingPathComponent("farend_singletalk_processed_coreml_128.wav")

    guard FileManager.default.fileExists(atPath: pythonFile.path) else {
      throw XCTSkip(
        "Python reference file not found at: \(pythonFile.path). Generate using python scripts or run swift test --filter RegenerateSamplesTests."
      )
    }
    guard FileManager.default.fileExists(atPath: coremlFile.path) else {
      throw XCTSkip(
        "CoreML output file not found at: \(coremlFile.path). Run swift test --filter RegenerateSamplesTests to generate sample files."
      )
    }

    let pythonSamples = try RegressionTestUtils.readWAVFile(pythonFile)
    let coremlSamples = try RegressionTestUtils.readWAVFile(coremlFile)

    let pythonRMS = RegressionTestUtils.computeRMS(pythonSamples)
    let coremlRMS = RegressionTestUtils.computeRMS(coremlSamples)
    let rmsRatio = max(pythonRMS, coremlRMS) / max(min(pythonRMS, coremlRMS), 1e-10)

    print("\n[128-unit] Python vs CoreML Reference Test:")
    print("  Baseline version: \(try RegressionTestUtils.loadBaselines().version)")
    print("  Python samples: \(pythonSamples.count), RMS: \(String(format: "%.6f", pythonRMS))")
    print("  CoreML samples: \(coremlSamples.count), RMS: \(String(format: "%.6f", coremlRMS))")
    print(
      "  RMS ratio: \(String(format: "%.2f", rmsRatio))x (baseline: \(String(format: "%.2f", pythonRef.rms_ratio))x)"
    )

    // Verify against baseline thresholds
    XCTAssertLessThan(
      pythonRMS, pythonRef.max_acceptable_rms,
      "Python reference should achieve near-silence (RMS \(pythonRMS) > max \(pythonRef.max_acceptable_rms))"
    )

    XCTAssertLessThan(
      coremlRMS, pythonRef.max_acceptable_rms,
      "CoreML should achieve near-silence (RMS \(coremlRMS) > max \(pythonRef.max_acceptable_rms))")

    XCTAssertLessThan(
      rmsRatio, pythonRef.max_rms_ratio,
      "CoreML/Python RMS ratio \(rmsRatio) exceeds max \(pythonRef.max_rms_ratio)")

    // Verify no regression from baseline (with tolerance)
    let maxAllowedRatio = pythonRef.rms_ratio + tolerance.rms_ratio
    XCTAssertLessThanOrEqual(
      rmsRatio, maxAllowedRatio,
      "RMS ratio \(String(format: "%.2f", rmsRatio)) regressed from baseline \(String(format: "%.2f", pythonRef.rms_ratio)) (max allowed: \(String(format: "%.2f", maxAllowedRatio)))"
    )
  }

  /// Verify CoreML echo suppression effectiveness using AEC challenge sample
  func testEchoSuppressionEffectiveness() throws {
    let baseline = try RegressionTestUtils.baseline(for: "128-unit")
    guard let aecBaseline = baseline.aec_challenge else {
      throw XCTSkip("No AEC challenge baseline defined for 128-unit model")
    }
    let tolerance = try RegressionTestUtils.tolerance()

    let samplesDir = RegressionTestUtils.samplesDir()
    let micFile = samplesDir.appendingPathComponent("farend_singletalk_mic.wav")
    let coremlFile = samplesDir.appendingPathComponent("farend_singletalk_processed_coreml_128.wav")

    guard FileManager.default.fileExists(atPath: micFile.path) else {
      throw XCTSkip(
        "Mic input file not found at: \(micFile.path). Run swift test --filter RegenerateSamplesTests to generate sample files."
      )
    }
    guard FileManager.default.fileExists(atPath: coremlFile.path) else {
      throw XCTSkip(
        "CoreML output file not found at: \(coremlFile.path). Run swift test --filter RegenerateSamplesTests to generate sample files."
      )
    }

    let micSamples = try RegressionTestUtils.readWAVFile(micFile)
    let coremlSamples = try RegressionTestUtils.readWAVFile(coremlFile)

    let micRMS = RegressionTestUtils.computeRMS(micSamples)
    let coremlRMS = RegressionTestUtils.computeRMS(coremlSamples)
    let reductionDB = RegressionTestUtils.computeReductionDB(micRMS, coremlRMS)

    print("\n[128-unit] Echo Suppression Test:")
    print("  Baseline version: \(try RegressionTestUtils.loadBaselines().version)")
    print("  Mic samples: \(micSamples.count)")
    print("  Output samples: \(coremlSamples.count)")
    print("  Mic RMS: \(String(format: "%.6f", micRMS))")
    print(
      "  Output RMS: \(String(format: "%.6f", coremlRMS)) (baseline: \(String(format: "%.6f", aecBaseline.output_rms)))"
    )
    print(
      "  Reduction: \(String(format: "%.1f", reductionDB)) dB (baseline: \(String(format: "%.1f", aecBaseline.reduction_db)) dB)"
    )

    // Detailed analysis
    let outputAnalysis = RegressionTestUtils.analyzeAudio(coremlSamples)
    outputAnalysis.printReport(label: "Output")

    // Verify output quality
    let hasNaN = coremlSamples.contains { $0.isNaN }
    let hasInf = coremlSamples.contains { $0.isInfinite }

    print("\n  Has NaN: \(hasNaN)")
    print("  Has Inf: \(hasInf)")

    XCTAssertFalse(hasNaN, "Output should not contain NaN values")
    XCTAssertFalse(hasInf, "Output should not contain infinite values")
    XCTAssertLessThanOrEqual(
      outputAnalysis.peakAmplitude, 1.0, "Output should be normalized (-1 to 1)")

    // Verify no regression from baseline (with tolerance)
    let minAllowedReduction = aecBaseline.reduction_db - tolerance.db
    XCTAssertGreaterThanOrEqual(
      reductionDB, minAllowedReduction,
      "Echo reduction \(String(format: "%.1f", reductionDB)) dB regressed from baseline \(String(format: "%.1f", aecBaseline.reduction_db)) dB (min allowed: \(String(format: "%.1f", minAllowedReduction)) dB)"
    )
  }

  /// Test that reprocessing the same input produces consistent output
  func testOutputConsistency() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .small)
    try processor.loadModelsFromPackage()

    let numSamples = 16000  // 1 second
    var farEnd = [Float](repeating: 0, count: numSamples)
    var nearEnd = [Float](repeating: 0, count: numSamples)

    // Generate test signal
    for i in 0..<numSamples {
      let t = Float(i) / 16000
      farEnd[i] = 0.3 * sin(2 * .pi * 440 * t)
      nearEnd[i] = 0.2 * sin(2 * .pi * 440 * t + 0.5)  // Echo with phase shift
    }

    // Process first time
    processor.feedFarEnd(farEnd)
    let output1 = processor.processNearEnd(nearEnd)

    // Reset and process again
    processor.resetStates()
    processor.feedFarEnd(farEnd)
    let output2 = processor.processNearEnd(nearEnd)

    // Outputs should be identical
    let correlation = RegressionTestUtils.computeCorrelation(output1, output2)
    print("\n[128-unit] Output Consistency Test:")
    print("  Correlation between runs: \(String(format: "%.6f", correlation))")

    // Relaxed threshold to account for floating-point variations across hardware
    XCTAssertGreaterThan(
      correlation, 0.9999,
      "Processing same input twice should produce nearly identical output (correlation > 0.9999)")
  }
}

// MARK: - 512-Unit Model Regression Tests

/// Regression tests for the 512-unit (large) model
final class RegressionTests512: XCTestCase {

  /// Verify CoreML echo suppression effectiveness using AEC challenge sample
  func testEchoSuppressionEffectiveness() throws {
    let baseline = try RegressionTestUtils.baseline(for: "512-unit")
    guard let aecBaseline = baseline.aec_challenge else {
      throw XCTSkip("No AEC challenge baseline defined for 512-unit model")
    }
    let tolerance = try RegressionTestUtils.tolerance()

    let samplesDir = RegressionTestUtils.samplesDir()
    let micFile = samplesDir.appendingPathComponent("farend_singletalk_mic.wav")
    let coremlFile = samplesDir.appendingPathComponent("farend_singletalk_processed_coreml_512.wav")

    guard FileManager.default.fileExists(atPath: micFile.path) else {
      throw XCTSkip(
        "Mic input file not found at: \(micFile.path). Run swift test --filter RegenerateSamplesTests to generate sample files."
      )
    }
    guard FileManager.default.fileExists(atPath: coremlFile.path) else {
      throw XCTSkip(
        "CoreML 512-unit output file not found at: \(coremlFile.path). Run swift test --filter RegenerateSamplesTests to generate sample files."
      )
    }

    let micSamples = try RegressionTestUtils.readWAVFile(micFile)
    let coremlSamples = try RegressionTestUtils.readWAVFile(coremlFile)

    let micRMS = RegressionTestUtils.computeRMS(micSamples)
    let coremlRMS = RegressionTestUtils.computeRMS(coremlSamples)
    let reductionDB = RegressionTestUtils.computeReductionDB(micRMS, coremlRMS)

    print("\n[512-unit] Echo Suppression Test:")
    print("  Baseline version: \(try RegressionTestUtils.loadBaselines().version)")
    print("  Mic samples: \(micSamples.count)")
    print("  Output samples: \(coremlSamples.count)")
    print("  Mic RMS: \(String(format: "%.6f", micRMS))")
    print(
      "  Output RMS: \(String(format: "%.6f", coremlRMS)) (baseline: \(String(format: "%.6f", aecBaseline.output_rms)))"
    )
    print(
      "  Reduction: \(String(format: "%.1f", reductionDB)) dB (baseline: \(String(format: "%.1f", aecBaseline.reduction_db)) dB)"
    )

    // Detailed analysis
    let outputAnalysis = RegressionTestUtils.analyzeAudio(coremlSamples)
    outputAnalysis.printReport(label: "Output")

    // Verify output quality
    let hasNaN = coremlSamples.contains { $0.isNaN }
    let hasInf = coremlSamples.contains { $0.isInfinite }

    print("\n  Has NaN: \(hasNaN)")
    print("  Has Inf: \(hasInf)")

    XCTAssertFalse(hasNaN, "Output should not contain NaN values")
    XCTAssertFalse(hasInf, "Output should not contain infinite values")
    XCTAssertLessThanOrEqual(
      outputAnalysis.peakAmplitude, 1.0, "Output should be normalized (-1 to 1)")

    // Verify no regression from baseline (with tolerance)
    let minAllowedReduction = aecBaseline.reduction_db - tolerance.db
    XCTAssertGreaterThanOrEqual(
      reductionDB, minAllowedReduction,
      "Echo reduction \(String(format: "%.1f", reductionDB)) dB regressed from baseline \(String(format: "%.1f", aecBaseline.reduction_db)) dB (min allowed: \(String(format: "%.1f", minAllowedReduction)) dB)"
    )
  }

  /// Test that reprocessing the same input produces consistent output
  func testOutputConsistency() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .large)
    try processor.loadModelsFromPackage()

    let numSamples = 16000  // 1 second
    var farEnd = [Float](repeating: 0, count: numSamples)
    var nearEnd = [Float](repeating: 0, count: numSamples)

    // Generate test signal
    for i in 0..<numSamples {
      let t = Float(i) / 16000
      farEnd[i] = 0.3 * sin(2 * .pi * 440 * t)
      nearEnd[i] = 0.2 * sin(2 * .pi * 440 * t + 0.5)  // Echo with phase shift
    }

    // Process first time
    processor.feedFarEnd(farEnd)
    let output1 = processor.processNearEnd(nearEnd)

    // Reset and process again
    processor.resetStates()
    processor.feedFarEnd(farEnd)
    let output2 = processor.processNearEnd(nearEnd)

    // Outputs should be identical
    let correlation = RegressionTestUtils.computeCorrelation(output1, output2)
    print("\n[512-unit] Output Consistency Test:")
    print("  Correlation between runs: \(String(format: "%.6f", correlation))")

    // Relaxed threshold to account for floating-point variations across hardware
    XCTAssertGreaterThan(
      correlation, 0.9999,
      "Processing same input twice should produce nearly identical output (correlation > 0.9999)")
  }
}

// MARK: - 256-Unit Model Regression Tests

/// Regression tests for the 256-unit (medium) model
final class RegressionTests256: XCTestCase {

  /// Verify CoreML echo suppression effectiveness using AEC challenge sample
  func testEchoSuppressionEffectiveness() throws {
    let samplesDir = RegressionTestUtils.samplesDir()
    let micFile = samplesDir.appendingPathComponent("farend_singletalk_mic.wav")
    let coremlFile = samplesDir.appendingPathComponent("farend_singletalk_processed_coreml_256.wav")

    guard FileManager.default.fileExists(atPath: micFile.path) else {
      throw XCTSkip(
        "Mic input file not found at: \(micFile.path). Run swift test --filter RegenerateSamplesTests to generate sample files."
      )
    }

    let micSamples = try RegressionTestUtils.readWAVFile(micFile)

    // If no pre-generated sample, process in streaming chunks
    let coremlSamples: [Float]
    if FileManager.default.fileExists(atPath: coremlFile.path) {
      coremlSamples = try RegressionTestUtils.readWAVFile(coremlFile)
    } else {
      print("\n[256-unit] Processing AEC challenge sample in-place (streaming)...")
      let loopbackFile = samplesDir.appendingPathComponent("farend_singletalk_lpb.wav")
      guard FileManager.default.fileExists(atPath: loopbackFile.path) else {
        throw XCTSkip(
          "Loopback file not found at: \(loopbackFile.path). Run swift test --filter RegenerateSamplesTests to generate sample files."
        )
      }
      let loopbackSamples = try RegressionTestUtils.readWAVFile(loopbackFile)

      let processor = DTLNAecEchoProcessor(modelSize: .medium)
      try processor.loadModelsFromPackage()

      // Process in streaming fashion with ~32ms chunks to simulate real-time processing
      let chunkSize = 512  // 32ms at 16kHz
      var output: [Float] = []
      var offset = 0
      while offset < min(micSamples.count, loopbackSamples.count) {
        let endOffset = min(offset + chunkSize, micSamples.count, loopbackSamples.count)
        let micChunk = Array(micSamples[offset..<endOffset])
        let loopbackChunk = Array(loopbackSamples[offset..<endOffset])

        processor.feedFarEnd(loopbackChunk)
        let processedChunk = processor.processNearEnd(micChunk)
        output.append(contentsOf: processedChunk)

        offset = endOffset
      }
      coremlSamples = output
    }

    let micRMS = RegressionTestUtils.computeRMS(micSamples)
    let coremlRMS = RegressionTestUtils.computeRMS(coremlSamples)
    let reductionDB = RegressionTestUtils.computeReductionDB(micRMS, coremlRMS)

    print("\n[256-unit] Echo Suppression Test:")
    print("  Mic samples: \(micSamples.count)")
    print("  Output samples: \(coremlSamples.count)")
    print("  Mic RMS: \(String(format: "%.6f", micRMS))")
    print("  Output RMS: \(String(format: "%.6f", coremlRMS))")
    print("  Reduction: \(String(format: "%.1f", reductionDB)) dB")

    // Detailed analysis
    let outputAnalysis = RegressionTestUtils.analyzeAudio(coremlSamples)
    outputAnalysis.printReport(label: "Output")

    // Verify output quality
    let hasNaN = coremlSamples.contains { $0.isNaN }
    let hasInf = coremlSamples.contains { $0.isInfinite }

    print("\n  Has NaN: \(hasNaN)")
    print("  Has Inf: \(hasInf)")

    XCTAssertFalse(hasNaN, "Output should not contain NaN values")
    XCTAssertFalse(hasInf, "Output should not contain infinite values")
    XCTAssertLessThanOrEqual(
      outputAnalysis.peakAmplitude, 1.0, "Output should be normalized (-1 to 1)")

    // 256-unit model should achieve significant echo reduction (>30 dB on this sample)
    XCTAssertGreaterThan(
      reductionDB, 30.0,
      "Echo reduction \(String(format: "%.1f", reductionDB)) dB should be > 30 dB for far-end singletalk"
    )
  }

  /// Test that reprocessing the same input produces consistent output
  func testOutputConsistency() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .medium)
    try processor.loadModelsFromPackage()

    let numSamples = 16000  // 1 second
    var farEnd = [Float](repeating: 0, count: numSamples)
    var nearEnd = [Float](repeating: 0, count: numSamples)

    // Generate test signal
    for i in 0..<numSamples {
      let t = Float(i) / 16000
      farEnd[i] = 0.3 * sin(2 * .pi * 440 * t)
      nearEnd[i] = 0.2 * sin(2 * .pi * 440 * t + 0.5)  // Echo with phase shift
    }

    // Process first time
    processor.feedFarEnd(farEnd)
    let output1 = processor.processNearEnd(nearEnd)

    // Reset and process again
    processor.resetStates()
    processor.feedFarEnd(farEnd)
    let output2 = processor.processNearEnd(nearEnd)

    // Outputs should be identical
    let correlation = RegressionTestUtils.computeCorrelation(output1, output2)
    print("\n[256-unit] Output Consistency Test:")
    print("  Correlation between runs: \(String(format: "%.6f", correlation))")

    // Relaxed threshold to account for floating-point variations across hardware
    XCTAssertGreaterThan(
      correlation, 0.9999,
      "Processing same input twice should produce nearly identical output (correlation > 0.9999)")
  }

  /// Test model loading
  func testModelLoading() throws {
    let processor = DTLNAecEchoProcessor(modelSize: .medium)

    XCTAssertFalse(processor.isInitialized, "Processor should not be initialized before loading")

    try processor.loadModelsFromPackage()

    XCTAssertTrue(processor.isInitialized, "Processor should be initialized after loading")
    XCTAssertEqual(processor.numUnits, 256, "Model should have 256 units")

    print("\n[256-unit] Model Loading Test:")
    print("  Model size: \(processor.modelSize)")
    print("  Units: \(processor.numUnits)")
    print("  Initialized: \(processor.isInitialized)")
  }
}
