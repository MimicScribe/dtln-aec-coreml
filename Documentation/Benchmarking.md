# Benchmarking Guide

Measure DTLN-aec performance on your target hardware.

## Quick Benchmark

Run the included benchmark tool:

```bash
# Default: 125 frames (1 second of audio)
swift run dtln-benchmark

# More iterations for stable results
swift run dtln-benchmark -n 1000

# JSON output for scripting
swift run dtln-benchmark --json
```

## Sample Output

```
DTLN-aec CoreML Benchmark
=========================
Device: MacBook Pro (Apple M1)
Frames per model: 500 (4.0 seconds of audio)

| Model | Params | Load    | Avg     | P99     | RT Ratio | Status |
|-------|--------|---------|---------|---------|----------|--------|
| 128   | 1.8M   |   421ms |  0.04ms |  0.74ms |   0.01x  | ✅     |
| 256   | 3.9M   |   430ms |  0.07ms |  1.25ms |   0.01x  | ✅     |
| 512   | 10.4M  |   512ms |  0.18ms |  2.97ms |   0.02x  | ✅     |

Real-time requirement: <8ms per frame
```

## Understanding Results

### Metrics

| Metric | Meaning |
|--------|---------|
| **Load** | Time to compile and load CoreML models |
| **Avg** | Average inference time per frame |
| **P99** | 99th percentile latency (worst-case) |
| **RT Ratio** | Processing time / frame duration (lower is better) |
| **Status** | ✅ if P99 < 8ms, ⚠️ if P99 < 16ms, ❌ otherwise |

### Real-Time Budget

Each frame represents 8ms of audio. To run in real-time:

- **P99 < 8ms**: Solid real-time performance
- **P99 < 16ms**: May work with buffering
- **P99 > 16ms**: Not suitable for real-time

### Neural Engine vs CPU

By default, CoreML uses `cpuAndNeuralEngine` which leverages the Neural Engine when available. To benchmark different configurations:

```swift
var config = DTLNAecConfig()
config.computeUnits = .cpuOnly  // Force CPU
// or
config.computeUnits = .all  // CPU + GPU + Neural Engine

let processor = DTLNAecEchoProcessor(config: config)
```

## CI Integration

Use JSON output for automated testing:

```bash
swift run dtln-benchmark --json > benchmark.json
```

Example GitHub Actions step:

```yaml
- name: Run benchmark
  run: |
    swift run -c release dtln-benchmark -n 100 --json > benchmark.json

- name: Check performance
  run: |
    # Extract P99 for 128-unit model
    p99=$(jq '.results[] | select(.model == "128") | .p99_ms' benchmark.json)
    if (( $(echo "$p99 > 8.0" | bc -l) )); then
      echo "::error::P99 latency ($p99 ms) exceeds 8ms budget"
      exit 1
    fi
```

## Performance Tips

### For Best Results

1. **Build in Release mode**: Debug builds are slower
   ```bash
   swift run -c release dtln-benchmark
   ```

2. **Warm up the Neural Engine**: First few inferences may be slower
   ```bash
   swift run dtln-benchmark -n 1000  # More frames = better average
   ```

3. **Close other apps**: Background processes affect latency

### Expected Performance

| Device | 128 units | 256 units | 512 units |
|--------|-----------|-----------|-----------|
| M1 Mac | <1ms P99 | <2ms P99 | <3ms P99 |
| M2 Mac | <1ms P99 | <1ms P99 | <2ms P99 |
| iPhone 14 | <1ms P99 | <2ms P99 | <3ms P99 |
| iPhone 12 | <1ms P99 | <2ms P99 | <3ms P99 |

*Actual results vary based on thermal state and system load.*

## Related

- [Audio Requirements](AudioRequirements.md)
- [API Reference](API.md)
