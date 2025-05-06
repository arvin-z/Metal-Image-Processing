# Metal Image Processor CLI

This project is a command-line tool for GPU-accelerated image processing using Apple's Metal framework. It allows users to apply various image filters and chain multiple filters together. It also has a benchmark function to demonstrate Metal performance against CPU.

## Features

* **Command-Line Interface:** Allows arguments specifying input/output files and processing operations.
* **Multiple Image Kernels:** Includes a selection of common image processing filters, mainly convolutional filters.
* **Filter Chaining:** Apply a sequence of multiple filters to an image in a single command. Intermediate results are kept on the GPU to maximize efficiency.
* **Benchmarking:** Provides a mode to benchmark the performance of the `gaussian_blur_3x3` filter (N iterations) on the GPU (Metal) against a CPU implementation.
* **GPU Acceleration:** Uses Metal framework for high-performance image processing on compatible Mac hardware (tested on Apple Silicon, might work on some AMD GPUs).

## Available Kernels

The following kernels can be used with the `-k` (`--kernels`) argument or for benchmarking:

* `grayscale`: Converts the image to grayscale.
* `gaussian_blur_3x3`: Applies a 3x3 Gaussian blur (convolutional).
* `sharpen_3x3`: Applies a 3x3 sharpening filter (convolutional).
* `edge_detect`: Applies a Sobel-based edge detection filter.

## Project Structure

* `GPUProject.xcodeproj`: Xcode project file.
* `main.cpp`: The main C++ source code for the CLI tool, including Metal setup, image handling, kernel execution logic, and argument parsing.
* `ProjectKernels.metal`: Contains the Metal Shading Language (MSL) code for all the image processing kernels.
* `lib/`:
    * `metal-cpp/`: Apple's C++ headers for the Metal framework. ([source](https://github.com/bkaradzic/metal-cpp))
    * `stb/`: Single-file public domain libraries (`stb_image.h` for loading, `stb_image_write.h` for saving images). ([source](https://github.com/nothings/stb))
    * `cxxopts/`: Open-source library for C++ command-line argument parsing. ([source](https://github.com/jarro2783/cxxopts))

## How to Bulid
1. Open `GPUProject.xcodeproj` in XCode.
2. In XCode, click `Product` on the menubar, then click `Build`. (Alternatively, `Cmd + B`)
3. The program executable (`ProcessImage`) and the metal library (`default.metallib`) will be saved to the project root directory (the same directory `GPUProject.xcodeproj` is in)
4. Open Terminal and `cd` to the project root directory.

## Command-Line Usage

`./MetalImageProcessor [options]`

Options:
* `-i <filepath>` or `--input <filepath>`: (Required) Path to the input image file (e.g., `input.png`).
* `-o <filepath>` or `--output <filepath>`: (Required) Path to save the processed output image (e.g., `output.png`).
* `-k <sequence>` or `--kernels <sequence>`: A comma-separated sequence of kernel names to apply.
    * Example: `-k gaussian_blur_3x3,sharpen_3x3` to run a blur and then a sharpen
    * If no argument, defaults to `grayscale`.
* `--benchmark <N>`: Activates benchmark mode. Runs `N` iterations of the `gaussian_blur_3x3` filter on both GPU (Metal) and CPU, then prints timing results. The output image will be the result of N applications of the GPU filter.
    * Example: `--benchmark 100`
    * Note: do not use the `-k` flag if using `--benchmark`
* `-h, --help`: Displays usage information and a list of available kernels.

#### Examples

(assumes you have input.png in the same directory)

Apply a single grayscale filter:

`./ProcessImage -i input.png -o output_gray.png -k grayscale`

Apply a grayscale filter, and then a blur filter:

`./ProcessImage -i input.png -o output_gray_blur.png -k grayscale,gaussian_blur_3x3`

Run a benchmark with 5 iterations:

`./ProcessImage -i input.png -o bench_out.png --benchmark 5`

Here is an expected output from the above benchmark command:
```
Starting Metal Image Processor...
Benchmark Mode: Applying 'gaussian_blur_3x3' 5 times.
Starting Metal Image Processor...
  Input:  input.png
  Output: bench_out.png
Using Metal Device: Apple M4 Max
Loaded input image 'input.png' (640x512).
Created Metal processing textures A and B on device.
Uploaded initial image data to texture A.
Created sampler state.
  Loading default Metal library...
  Loaded default Metal library.
  Looking for kernel function: k_convolve...
  Found kernel function: k_convolve
  Creating compute pipeline state...
  Compute pipeline state created successfully.
  Created and cached PSO for 'k_convolve'.
  Dispatching Metal kernel: Grid(640x512), Threadgroup(32x32)
  Kernel submitted to GPU. Waiting for completion...
  GPU execution finished.
  Using cached PSO for 'k_convolve'.
  Dispatching Metal kernel: Grid(640x512), Threadgroup(32x32)
  Kernel submitted to GPU. Waiting for completion...
  GPU execution finished.
  Using cached PSO for 'k_convolve'.
  Dispatching Metal kernel: Grid(640x512), Threadgroup(32x32)
  Kernel submitted to GPU. Waiting for completion...
  GPU execution finished.
  Using cached PSO for 'k_convolve'.
  Dispatching Metal kernel: Grid(640x512), Threadgroup(32x32)
  Kernel submitted to GPU. Waiting for completion...
  GPU execution finished.
  Using cached PSO for 'k_convolve'.
  Dispatching Metal kernel: Grid(640x512), Threadgroup(32x32)
  Kernel submitted to GPU. Waiting for completion...
  GPU execution finished.

Execution of filter sequence complete.
Downloaded processed data.
Saved output image to: 'bench_out.png'.

--- Benchmark Results for 5 applications of 'gaussian_blur_3x3' ---
GPU Total Time (upload + 5 kernels + download): 30 ms
Running CPU benchmark...
CPU Total Time (5 kernels): 358 ms
Approx Average Time per iteration: GPU ~= 6 ms, CPU = 71.6 ms
CPU / GPU (Total Time Ratio): 11.9333
Execution finished successfully.
```

#### Example Output Images

Grayscale:

![input image](https://github.com/arvin-z/Metal-Image-Processing/blob/d6a64fe1e1bf25602632d8e635f0406d98ef03c5/sample_outputs/output_1.png)

Blur filter:

![input image](https://github.com/arvin-z/Metal-Image-Processing/blob/d6a64fe1e1bf25602632d8e635f0406d98ef03c5/sample_outputs/output_1c.png)

Sharpen filter:

![input image](https://github.com/arvin-z/Metal-Image-Processing/blob/d6a64fe1e1bf25602632d8e635f0406d98ef03c5/sample_outputs/output_1d.png)

Edge detection filter:

![input image](https://github.com/arvin-z/Metal-Image-Processing/blob/d6a64fe1e1bf25602632d8e635f0406d98ef03c5/sample_outputs/output_1b.png)


