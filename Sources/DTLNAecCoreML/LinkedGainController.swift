// LinkedGainController.swift
// Linked gain control for DTLN-AEC preprocessing
//
// Applies identical gain to both mic and system audio to preserve
// the echo-to-reference ratio critical for DTLN convergence stability.

import Accelerate
import Foundation

/// Linked gain controller for DTLN preprocessing.
///
/// Applies identical gain to both mic and system audio to preserve the echo relationship.
/// Uses soft-knee compression with dual-path envelope tracking for smooth gain changes.
///
/// - Note: Not Sendable due to mutable state; thread safety is provided by
///   DTLNAecEchoProcessor's lock when used within the processor.
public final class LinkedGainController {

  // MARK: - Soft-knee Parameters

  /// Compression threshold (default: 0.5, approximately -6dB)
  public let threshold: Float

  /// Soft-knee width for gradual transition (default: 0.3)
  public let kneeWidth: Float

  /// Compression ratio above threshold (default: 8.0)
  public let ratio: Float

  // MARK: - Gain Limits

  /// Minimum allowed gain (-20dB floor, default: 0.1)
  public let minGain: Float

  /// Maximum allowed gain (+6dB ceiling, default: 2.0)
  public let maxGain: Float

  // MARK: - Attack/Release Coefficients

  /// Fast attack coefficient (~1ms equivalent for 8ms frames)
  private let attackCoeff: Float

  /// Slow release coefficient (~100ms equivalent for 8ms frames)
  private let releaseCoeff: Float

  // MARK: - State

  /// Fast envelope follower (instant attack)
  private var fastEnvelope: Float = 0.0

  /// Slow envelope follower (smoothed release)
  private var slowEnvelope: Float = 0.0

  /// Current smoothed gain value
  private var currentGain: Float = 1.0

  // MARK: - Initialization

  /// Creates a linked gain controller with specified parameters.
  ///
  /// - Parameters:
  ///   - threshold: Compression threshold (default: 0.5)
  ///   - kneeWidth: Soft-knee width (default: 0.3)
  ///   - ratio: Compression ratio (default: 8.0)
  ///   - minGain: Minimum gain floor (default: 0.1)
  ///   - maxGain: Maximum gain ceiling (default: 2.0)
  ///   - attackMs: Attack time in milliseconds (default: 1.0)
  ///   - releaseMs: Release time in milliseconds (default: 100.0)
  ///   - frameMs: Frame duration in milliseconds (default: 8.0 for DTLN's 128-sample frames)
  public init(
    threshold: Float = 0.5,
    kneeWidth: Float = 0.3,
    ratio: Float = 8.0,
    minGain: Float = 0.1,
    maxGain: Float = 2.0,
    attackMs: Float = 1.0,
    releaseMs: Float = 100.0,
    frameMs: Float = 8.0
  ) {
    self.threshold = threshold
    self.kneeWidth = kneeWidth
    self.ratio = ratio
    self.minGain = minGain
    self.maxGain = maxGain

    // Compute per-frame coefficients
    // For ~1ms attack with 8ms frames: nearly instant
    self.attackCoeff = 1.0 - exp(-frameMs / attackMs)
    // For ~100ms release with 8ms frames: gradual decay
    self.releaseCoeff = 1.0 - exp(-frameMs / releaseMs)
  }

  // MARK: - Public API

  /// Computes the linked gain to apply to both mic and system audio.
  ///
  /// Uses the maximum of mic and system peaks to determine gain, ensuring
  /// both streams are scaled identically to preserve the echo relationship.
  ///
  /// - Parameters:
  ///   - micPeak: Peak absolute value of mic audio frame
  ///   - sysPeak: Peak absolute value of system/loopback audio frame
  /// - Returns: Gain value to apply to both streams (clamped to [minGain, maxGain])
  public func computeLinkedGain(micPeak: Float, sysPeak: Float) -> Float {
    // Use maximum peak from both streams
    let inputPeak = max(micPeak, sysPeak)

    // Update envelope followers
    updateEnvelopes(inputPeak: inputPeak)

    // Use max(fast, slow) for fast attack + slow release behavior
    let envelope = max(fastEnvelope, slowEnvelope)

    // Compute target gain from soft-knee curve
    let targetGain = computeSoftKneeGain(envelope: envelope)

    // Smooth gain changes to avoid audible artifacts
    let gainDelta = targetGain - currentGain
    if gainDelta < 0 {
      // Attacking (reducing gain) - use fast coefficient
      currentGain += gainDelta * attackCoeff
    } else {
      // Releasing (increasing gain) - use slow coefficient
      currentGain += gainDelta * releaseCoeff
    }

    // Clamp to valid range
    return max(minGain, min(maxGain, currentGain))
  }

  /// Resets the envelope and gain state.
  ///
  /// Call when starting a new recording session to clear accumulated state.
  public func reset() {
    fastEnvelope = 0.0
    slowEnvelope = 0.0
    currentGain = 1.0
  }

  /// Captures the current envelope and gain state.
  ///
  /// Use with `restoreState(_:)` to preserve state across operations that
  /// would otherwise corrupt it (e.g., processing zero-padded frames in flush).
  ///
  /// - Returns: A tuple containing the current fast envelope, slow envelope, and gain values.
  public func captureState() -> (fastEnvelope: Float, slowEnvelope: Float, currentGain: Float) {
    return (fastEnvelope, slowEnvelope, currentGain)
  }

  /// Restores previously captured state.
  ///
  /// - Parameter state: State tuple previously returned from `captureState()`.
  public func restoreState(_ state: (fastEnvelope: Float, slowEnvelope: Float, currentGain: Float))
  {
    fastEnvelope = state.fastEnvelope
    slowEnvelope = state.slowEnvelope
    currentGain = state.currentGain
  }

  // MARK: - Private Helpers

  /// Updates the dual-path envelope followers.
  private func updateEnvelopes(inputPeak: Float) {
    // Fast path: instant attack for transients
    if inputPeak > fastEnvelope {
      fastEnvelope = inputPeak
    } else {
      fastEnvelope += (inputPeak - fastEnvelope) * releaseCoeff
    }

    // Slow path: EMA smoothed for natural release
    slowEnvelope += (inputPeak - slowEnvelope) * releaseCoeff
  }

  /// Computes gain using soft-knee compression curve.
  ///
  /// Curve behavior:
  /// - Below (threshold - kneeWidth/2): Unity gain (1.0)
  /// - Within knee region: Quadratic transition
  /// - Above (threshold + kneeWidth/2): Full compression (ratio:1)
  private func computeSoftKneeGain(envelope: Float) -> Float {
    let halfKnee = kneeWidth / 2.0

    // Below knee: unity gain
    if envelope <= threshold - halfKnee {
      return 1.0
    }

    // Above knee: full compression
    if envelope >= threshold + halfKnee {
      // Output = threshold + (input - threshold) / ratio
      // Gain = output / input
      let outputLevel = threshold + (envelope - threshold) / ratio
      return outputLevel / max(envelope, 1e-6)
    }

    // Within soft knee: quadratic interpolation
    // x represents position within knee region normalized to [0, kneeWidth]
    let x = envelope - (threshold - halfKnee)
    let normalizedX = x / kneeWidth

    // Quadratic blend: starts at unity, ends at full compression
    // compressionAmount goes from 0 to 1 across the knee
    let compressionAmount = normalizedX * normalizedX

    // Early exit: if barely in knee region, return unity gain
    if compressionAmount < 1e-4 {
      return 1.0
    }

    // At the end of the knee, what would full compression give us?
    let fullCompressionGain: Float
    if envelope > 1e-6 {
      let outputLevel = threshold + (envelope - threshold) / ratio
      fullCompressionGain = outputLevel / envelope
    } else {
      fullCompressionGain = 1.0
    }

    // Blend between unity and full compression
    return 1.0 + compressionAmount * (fullCompressionGain - 1.0)
  }
}
