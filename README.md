# Metal Image Processing

This project aims to implement an image processing pipeline using Metal framework. It should provide GPU acceleration on Apple Silicon Macs (and theoretically certain Intel/AMD GPU Macs)

## Current Demo Version

The current version provides a basic demonstration by taking an `input.png` file, running a simple grayscale kernel, and saving the grayscale image to `output.png`.

### Contents/structure:
- `GPUProject.xcodeproj` Xcode project data
- `GPUProject` program and kernel code:
  - `main.cpp` program code (C++)
  - `ProjectKernels.metal` kernel code (Metal Shading Language, based on C++)
- `lib` contains `metal-cpp` (the Metal interface for C++), as well as `stb` libraries for loading/saving image files.


### Steps to run the program:

(Tested on macOS 15.4, Xcode 16.3)

1. Open `GPUProject.xcodeproj` in Xcode
2. Click Run to build and run the program (assuming input.png is in the project root directory)
3. An `output.png` with the grayscale kernel applied will be saved in the project root directory

Expected output:
```
Starting preliminary Metal Image Processor...
Using Metal Device: Apple M4 Max
Loaded input image 'input.png' (640x512).
Created Metal input and output textures on device.
Uploaded image data to input texture.
  Loading default Metal library...
  Loaded default Metal library.
  Looking for kernel function: k_grayscale...
  Found kernel function: k_grayscale
  Creating compute pipeline state...
  Compute pipeline state created successfully.
Setup compute pipeline state for kernel: 'k_grayscale'.
  Dispatching Metal kernel: Grid(640x512), Threadgroup(32x32)
  Kernel submitted to GPU. Waiting for completion...
  GPU execution finished.
Metal kernel execution complete.
Downloaded processed data from output texture.
Saved output image to: 'output.png'.
Processing complete...
Execution finished successfully.
Program ended with exit code: 0
```

Before and after:


![input image](https://cdn.discordapp.com/attachments/307290656286441473/1363043086040826007/input.png?ex=6804984b&is=680346cb&hm=aef6ae5e76cae2f53cc476b75eb4f8705b3509fac80e8aa39ee86e8db0c540ee&) ![output image](https://cdn.discordapp.com/attachments/307290656286441473/1363043086363529337/output.png?ex=6804984b&is=680346cb&hm=a71d2a56bfc23d0a2b3b81afcf1533ea6798316376dabce259b1deddd94fb404&)

