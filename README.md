# comfyui-benchmark
Custom extension for benchmarking ComfyUI performance

## Output

Benchmark results are saved to the `ComfyUI/outputs/benchmarks` directory as JSON files with timestamps in their filenames.

### What gets captured

A single background sampler runs on **every** backend and records, on a fixed interval (`check_interval`):

- **CPU + RAM** (`psutil_data`) — `timestamp, ram_used_bytes, ram_total_bytes, cpu_percent`. Captured on all machines, including CPU-only.
- **GPU — NVIDIA** (`nvidia_smi_data`) — VRAM used/total, GPU + memory utilization %, temperature, power draw/instant/limit, SM + memory clocks, PCIe link gen/width (via `nvidia-smi`).
- **GPU — Apple Silicon / MPS** (`device_memory_data`) — `timestamp, current_allocated_bytes` (bytes held by live tensors) and `driver_allocated_bytes` (total bytes the Metal driver reserved), via `torch.mps` (there is no `nvidia-smi` on Apple Silicon).

Alongside the time series, each run records **operation timings** (model load, CLIP, VAE encode/decode, sampler total + per-step iteration times → it/s, total workflow time) and a static **`device_info`** snapshot: GPU name, `backend` (`nvidia` / `mps` / `cpu`), PyTorch + Python + ComfyUI versions, OS/platform/architecture, `cpu_model` + physical/logical core counts, and total VRAM/RAM.

`device_info.backend` exists so results are **never compared across memory types** — MPS/CPU memory is drawn from unified system RAM, not dedicated VRAM. (Timeline visualization of the MPS series in `visualize_benchmark.py` is a follow-up; all of the above is captured now.)

## Configuration

The benchmark behavior can be configured using `config.yaml`. If a Benchmark Workflow node is placed on the graph, `file_prefix` and `file_postfix` are prepended and appended to benchmark file name, respectively.

### Setting up config.yaml

- On first run, a `config.yaml` file is automatically generated with default settings
- Alternatively, you can rename `config.yaml.example` to `config.yaml` before the first run to use your preferred settings
- The configuration is read at the start of each workflow execution, so you can modify settings without restarting ComfyUI

### Configuration Options

The `config.yaml` file contains the following settings to control benchmark data collection:

- `iteration_times` (default: true) - When set to false, disables logging of individual step execution times
- `enable_thread` (default: true) - When set to false, disables the background sampler (no CPU/RAM time series, and no GPU VRAM/utilization/power/memory data). Operation timings are still recorded. (The `nvidia-smi` config key is retained for backward compatibility; the sampler now covers all backends.)
- `check_interval` (default: 0.25) - The time interval in seconds between sampler ticks during workflow execution
- `require_node` (default: false) - When set to true, benchmarks are only saved if a Benchmark Workflow node is present on the graph AND `capture_benchmark` is set to true.

## Visualization

The benchmark results can be visualized using the included `visualize_benchmark.py` script.

<img width="3840" height="2160" alt="use-in-readmy" src="https://github.com/user-attachments/assets/7f71f195-5032-4280-9494-2501237b98cc" />


### Usage

1. Install the required dependency:
   ```bash
   pip install -r visualize_requirements.txt
   ```

2. Run the visualization script with a benchmark file:
   ```bash
   python visualize_benchmark.py <benchmark_file.json>
   ```
   
   For example:
   ```bash
   python visualize_benchmark.py benchmark_20250904_140902.json
   ```

   The benchmark file can be specified with any path - it doesn't need to follow a specific naming format.

3. **Comparing Multiple Benchmarks**: To compare multiple benchmark results, you have several options:
   
   Provide multiple JSON files:
   ```bash
   python visualize_benchmark.py benchmark1.json benchmark2.json benchmark3.json
   ```
   
   Compare all benchmarks in a directory:
   ```bash
   # Current directory
   python visualize_benchmark.py .
   
   # Specific directory
   python visualize_benchmark.py /path/to/benchmarks/
   ```
   
   Mix directories and individual files:
   ```bash
   python visualize_benchmark.py benchmarks_dir/ extra_benchmark.json
   ```
   
   This creates a comparison view showing only the Workflow Operations Timeline for each benchmark, making it easy to compare performance across different runs or configurations. When providing a directory, the script automatically finds and validates all JSON files with proper benchmark structure.

### Command-line Arguments

The visualization script supports the following optional arguments:

- `--save-html [PATH]` - Save the visualization as an interactive HTML file. If no path is provided, saves in the current directory with `_visualization.html` suffix.
- `--save-image [PATH]` - Save the visualization as a static image (PNG, JPG, SVG, etc.). If no path is provided, saves in the current directory with `_visualization.png` suffix. **Note: Requires the `kaleido` package (`pip install kaleido`).**
- `--no-show` - Skip opening the visualization in the browser (useful for batch processing or headless environments).

Examples:
```bash
# Open in browser and save as HTML in current directory
python visualize_benchmark.py benchmark_20250904_140902.json --save-html

# Save as PNG with custom path without opening browser
python visualize_benchmark.py benchmark_20250904_140902.json --save-image results/benchmark.png --no-show

# Save both HTML and image formats
python visualize_benchmark.py benchmark_20250904_140902.json --save-html report.html --save-image report.svg

# Compare multiple benchmarks and save as HTML
python visualize_benchmark.py benchmark1.json benchmark2.json --save-html comparison.html

# Compare all benchmarks in current directory
python visualize_benchmark.py . --save-html all_benchmarks.html

# Compare benchmarks from a specific directory
python visualize_benchmark.py ./outputs/benchmarks/ --save-image comparison.png
```

### Visualization Features

The script creates an interactive Plotly graph showing:

**Single Benchmark Mode:**
- **Device Information Table** - GPU details, PyTorch version, startup arguments
- **VRAM Usage** - Timeline showing memory consumption with initial and maximum VRAM lines (NVIDIA only*)
- **RAM Usage** - System memory consumption over time (when psutil data is available)
- **GPU Utilization** - GPU usage percentage over time (NVIDIA only*)
- **Power Usage** - Power draw with both average and instantaneous values (NVIDIA only*)
- **Workflow Operations Timeline** - Color-coded bars showing when different operations occurred (model loading, sampling, VAE encode/decode, etc.) with total workflow duration indicator

**Comparison Mode (Multiple Benchmarks):**
- **Stacked Workflow Operations Timelines** - Each benchmark's operations are displayed in a separate row with:
  - Shared time axis for easy comparison
  - Total duration indicators for each benchmark
  - Consistent color coding across all benchmarks
  - Aligned to the longest benchmark's duration

*Note: VRAM usage, GPU utilization, and power usage graphs are only available on NVIDIA devices with the `nvidia-smi` command working. The workflow operations timeline will still be displayed for all devices.

The visualization opens in your default web browser and supports interactive zooming, panning, and hovering for detailed information.
