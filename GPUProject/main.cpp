//
//  main.cpp
//  GPUProject
//
//  GPU image processing using Apple's Metal framework via the metal-cpp wrapper.
//  This preliminary version loads an image, applies a grayscale compute kernel on the GPU, and saves the result.
//

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <memory>
#include <filesystem>
#include <map>
#include <variant>
#include <algorithm>
#include <sstream>
#include <chrono>     // For timing


// stb (external C library for simple image loading/saving)
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

// === Metal Framework (via metal-cpp) ===
// metal-cpp provides C++ headers for the underlying Objective-C Metal framework.
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp> // Basic types used by Metal (strings, errors, etc.)
#include <Metal/Metal.hpp>           // Core Metal API (devices, commands, buffers, textures, shaders)
#include <QuartzCore/QuartzCore.hpp> // Provides display integration (not used in this project)

// cxxopts for command line parsing
#include <cxxopts.hpp>


// KERNEL INFO
// struct to hold data needed for convolution kernels
struct ConvolutionInfo {
    std::vector<float> matrix;
    int dimension;
    std::string metalKernelName = "k_convolve"; // Most convolutions use the generic kernel
};

// Define standard convolution matrices
const std::vector<float> GAUSSIAN_BLUR_3X3_MATRIX = {
    1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f,
    2.0f/16.0f, 4.0f/16.0f, 2.0f/16.0f,
    1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f
};

const std::vector<float> SHARPEN_3X3_MATRIX = {
     0.0f, -1.0f,  0.0f,
    -1.0f,  5.0f, -1.0f,
     0.0f, -1.0f,  0.0f
};

struct SimpleKernelInfo {
     std::string metalKernelName;
};

// Use std::variant to hold info for different kernel types
using KernelInfo = std::variant<SimpleKernelInfo, ConvolutionInfo>;

// Define the kernel name used for benchmarking
const std::string BENCHMARK_KERNEL_USER_NAME = "gaussian_blur_3x3";

// Map CLI kernel names to their info and Metal function names
std::map<std::string, KernelInfo> KERNEL_REGISTRY = {
    {"grayscale",         SimpleKernelInfo{"k_grayscale"}},
    {BENCHMARK_KERNEL_USER_NAME, ConvolutionInfo{GAUSSIAN_BLUR_3X3_MATRIX, 3}},
    {"sharpen_3x3",       ConvolutionInfo{SHARPEN_3X3_MATRIX, 3}},
    {"edge_detect",       SimpleKernelInfo{"k_edge_detect"}}
    // could add more
};

// Helper function to split string by delimiter
std::vector<std::string> split(const std::string& s, char delimiter) {
   std::vector<std::string> tokens;
   std::string token;
   std::istringstream tokenStream(s);
   while (std::getline(tokenStream, token, delimiter)) {
      // Remove leading/trailing whitespace from token if necessary
      size_t first = token.find_first_not_of(' ');
      if (std::string::npos == first) continue; // Skip empty tokens or tokens with only spaces
      size_t last = token.find_last_not_of(' ');
      tokens.push_back(token.substr(first, (last - first + 1)));
   }
   return tokens;
}


// === CPU Implementation for Gaussian Blur 3x3 ===
void cpuGaussianBlur3x3(std::vector<unsigned char>& imageData, int width, int height) {
    if (imageData.empty() || width <= 0 || height <= 0) {
        throw std::invalid_argument("Invalid image data for CPU Gaussian Blur.");
    }
    const int channels = 4;
    std::vector<unsigned char> tempImageData = imageData; // Work on a copy
    const auto& kernelMatrix = GAUSSIAN_BLUR_3X3_MATRIX;
    const int kernelDim = 3;
    const int kernelRadius = kernelDim / 2;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;
            for (int ky = -kernelRadius; ky <= kernelRadius; ++ky) {
                for (int kx = -kernelRadius; kx <= kernelRadius; ++kx) {
                    int pixelX = std::min(std::max(x + kx, 0), width - 1);
                    int pixelY = std::min(std::max(y + ky, 0), height - 1);
                    size_t neighborIndex = (static_cast<size_t>(pixelY) * width + pixelX) * channels;
                    float weight = kernelMatrix[(ky + kernelRadius) * kernelDim + (kx + kernelRadius)];
                    sumR += static_cast<float>(tempImageData[neighborIndex + 0]) * weight;
                    sumG += static_cast<float>(tempImageData[neighborIndex + 1]) * weight;
                    sumB += static_cast<float>(tempImageData[neighborIndex + 2]) * weight;
                }
            }
            size_t currentIndex = (static_cast<size_t>(y) * width + x) * channels;
            imageData[currentIndex + 0] = static_cast<unsigned char>(std::min(std::max(sumR, 0.0f), 255.0f));
            imageData[currentIndex + 1] = static_cast<unsigned char>(std::min(std::max(sumG, 0.0f), 255.0f));
            imageData[currentIndex + 2] = static_cast<unsigned char>(std::min(std::max(sumB, 0.0f), 255.0f));
            imageData[currentIndex + 3] = tempImageData[currentIndex + 3]; // Preserve alpha
        }
    }
}


// === Metal Memory Management EXplanation ===
// Metal uses Objective-C style reference counting for lifetime management.
// metal-cpp wraps this with C++ pointers:
// - NS::SharedPtr<T>: Similar to C++ shared_ptr. Manages the reference count automatically.
//                    When the last SharedPtr goes out of scope, the underlying Metal object is released.
// - NS::TransferPtr(T* ptr): Takes ownership of raw pointer and wraps it in NS::SharedPtr
// - NS::AutoreleasePool: A mechanism from Objective-C. Objects can be "autoreleased", meaning they
//                       are added to the current pool and will be released later when the pool
//                       is drained (destroyed). This is often used for temporary objects created
//                       within a scope, like command buffers or encoders



// this function finds and initializes the default Metal GPU device (i.e. the built-in GPU)
NS::SharedPtr<MTL::Device> setupMetalDevice() {
    // MTL::CreateSystemDefaultDevice(): Asks Metal for the default GPU suitable for computation.
    // Returns a raw pointer to an MTL::Device object
    MTL::Device* pRawDevice = MTL::CreateSystemDefaultDevice();
    if (!pRawDevice) {
        throw std::runtime_error("Failed to get default Metal device. Metal may not be supported.");
    }

    // NS::TransferPtr(): Takes ownership of the raw pointer returned by CreateSystemDefaultDevice.
    // It wraps the raw MTL::Device* in an NS::SharedPtr<MTL::Device>, which will handle releasing
    // the device automatically when the SharedPtr goes out of scope
    NS::SharedPtr<MTL::Device> pDevice = NS::TransferPtr(pRawDevice);

    std::cout << "Using Metal Device: " <<
        // pDevice->name() returns an NS::String*, access its C-string representation for printing
        (pDevice->name() ? pDevice->name()->utf8String() : "Unknown") << std::endl;

    // The SharedPtr is returned, transferring ownership to the caller (main function).
    return pDevice;
}


// This function loads an image from a file into CPU buffer
void loadImageData(const std::string& inputPath,
                   int& width,
                   int& height,
                   std::vector<unsigned char>& imageData)
{
    int channels = 0; // Original channels in file
    // stbi_load forces 4 channels (RGBA). Returns raw C pointer to pixel data.
    unsigned char* loadedPixels = stbi_load(inputPath.c_str(), &width, &height, &channels, 4);
    if (!loadedPixels) {
        throw std::runtime_error("Failed to load image file: " + inputPath + " - Reason: " + stbi_failure_reason());
    }

    // Calculate size and copy data into a std::vector
    size_t imageSize = static_cast<size_t>(width) * height * 4; // 4 bytes per pixel (RGBA)
    imageData.resize(imageSize);
    std::copy(loadedPixels, loadedPixels + imageSize, imageData.begin());

    // Free the buffer allocated by stbi_load.
    stbi_image_free(loadedPixels);
}



// this function creates a Metal Texture object on the GPU (which will eventually hold the image data)
// the usage parameter specifies how the texture will be used by GPU (read, written, etc.)
// the storage parameter specifies how the texture is stored on memory.
//   - Apple silicon has unified RAM/VRAM, so using "Shared" storage mode simplifies transfers by
//     having it accessible by both CPU and GPU directly.
NS::SharedPtr<MTL::Texture> createMetalTexture(NS::SharedPtr<MTL::Device> pDevice, // Accepts SharedPtr
                                               int width,
                                               int height,
                                               MTL::PixelFormat format,
                                               MTL::TextureUsage usage,
                                               MTL::StorageMode storage = MTL::StorageModeShared) // Defaulting to Shared for simplicity on UMA
{
    // 1. Create a Texture Descriptor: This is a configuration object, not the texture itself.
    // Similar to setting up parameters before an API call in CUDA/OpenCL.
    // MTL::TextureDescriptor::texture2DDescriptor returns a raw pointer
    // the "false" parameter refers to whether or not the texture is mipmapepd
    MTL::TextureDescriptor* pRawDesc = MTL::TextureDescriptor::texture2DDescriptor(format, width, height, false);
    NS::SharedPtr<MTL::TextureDescriptor> pTextureDesc = NS::TransferPtr(pRawDesc);
    
    if (!pTextureDesc) { throw std::runtime_error("Failed to create Texture Descriptor."); }

    // Configure the descriptor with desired usage and storage mode.
    pTextureDesc->setUsage(usage);     // How the GPU kernel will access it (Read, Write, Both)
    pTextureDesc->setStorageMode(storage); // Memory residency and access (CPU/GPU)

    // 2. Create the Texture: Ask the GPU to create the actual texture resource based on the descriptor.
    MTL::Texture* pRawTexture = pDevice->newTexture(pTextureDesc.get());
    NS::SharedPtr<MTL::Texture> pTexture = NS::TransferPtr(pRawTexture);
    if (!pTexture) { throw std::runtime_error("Failed to create texture."); }

    // The SharedPtr holding the texture is returned.
    return pTexture;
}


// this function uploads the image data from the CPU buffer to the Metal Texture on the GPU
// similar to using cudaMemcpy2DtoArray in CUDA
void uploadImageDataToTexture(const std::vector<unsigned char>& imageData,
                              NS::SharedPtr<MTL::Texture> pTexture)
{
    if (!pTexture || imageData.empty()) {
        throw std::invalid_argument("Invalid texture or image data for upload");
    }
    
    // Get texture dimensions. NS::UInteger is Metal-compatible unsigned integer type.
    NS::UInteger width = pTexture->width();
    NS::UInteger height = pTexture->height();
    // Calculate bytes per row (pitch/stride). For RGBA its width * 4 bytes
    NS::UInteger bytesPerRow = width * 4;
    // Define the region of the texture to update (in this case the entire texture)
    MTL::Region region = MTL::Region::Make2D(0, 0, width, height); // thats from x=0, y=0

    // replaceRegion: Copies data from the CPU buffer (imageData.data()) to the specified texture region.
    // the important parameters for replaceRegion are:
    // - region: The 2D area within the texture to write to.
    // - imageData.data(): Raw pointer to the source CPU data.
    // - bytesPerRow: Stride of the source CPU data.
    pTexture->replaceRegion(region, 0, 0, static_cast<const void*>(imageData.data()), bytesPerRow, 0);
}


// this function loads and creates a Metal Compute Pipeline State object for our kernel function
// presently, thats our greyscale kernel form ProjectKernels.metal
// (Metal precompiles the kernels into a library, which we will load)
NS::SharedPtr<MTL::ComputePipelineState> setupPipelineState(NS::SharedPtr<MTL::Device> pDevice,
                                                            const std::string& kernelName)
{

    NS::Error* pRawError = nullptr;

    // --- 1. Load the Default Shader Library ---
    // Assumes Metal shader code (from .metla file) was compiled and linked into the app's Metal library
    // XCode does this automatically on build
    std::cout << "  Loading default Metal library..." << std::endl;
    MTL::Library* pRawLibrary = pDevice->newDefaultLibrary();
    NS::SharedPtr<MTL::Library> pDefaultLibrary = NS::TransferPtr(pRawLibrary);
    
    if (!pDefaultLibrary) {
        throw std::runtime_error("Failed to load default Metal library. Make sure .metal file is getting compiled/linked.");
    }
    std::cout << "  Loaded default Metal library." << std::endl;

    // --- 2. Get the Kernel Function from the Library ---
    // the kernel is identified by it's name, as a String
    NS::String* pKernelNsString = NS::String::string(kernelName.c_str(), NS::UTF8StringEncoding);

    // newFunction() searches the Metal library for the named function/kernel, and returns pointer
    std::cout << "  Looking for kernel function: " << kernelName << "..." << std::endl;
    MTL::Function* pRawFunction = pDefaultLibrary->newFunction(pKernelNsString);
    NS::SharedPtr<MTL::Function> pFunction = NS::TransferPtr(pRawFunction);

    if (!pFunction) {
        throw std::runtime_error("Failed to find kernel function: '" + kernelName + "' in the default library.");
    }
    std::cout << "  Found kernel function: " << kernelName << std::endl;

    // --- 3. Create the Compute Pipeline State (PSO) ---
    // Creates an object representing the compiled kernel ready for execution.
    // newComputePipelineState() returns a raw PSO pointer and potentially sets pRawError
    std::cout << "  Creating compute pipeline state..." << std::endl;
    MTL::ComputePipelineState* pRawPSO = pDevice->newComputePipelineState(pFunction.get(), &pRawError);
    NS::SharedPtr<MTL::ComputePipelineState> pPSO = NS::TransferPtr(pRawPSO);

    
    // If newComputePipelineState failed, pRawPSO will be null AND pRawError might point to an error object.
    NS::SharedPtr<NS::Error> pErrorWrapper; // Smart pointer to manage the error object if it exists.
    if (pRawError != nullptr) {
        // retain the error object, so that it can't be deallocated too early
        std::cout << "  Capturing PSO creation error details..." << std::endl;
        pErrorWrapper = NS::TransferPtr(pRawError->retain());
    }
    // Check if PSO creation failed
    if (!pPSO) {
        std::string errorMsg = "Failed to create compute pipeline state for kernel '" + kernelName + "'.";
        if (pErrorWrapper) {
            // Access  error description
            errorMsg += " Error: " + std::string(pErrorWrapper->localizedDescription()->utf8String());
        } else {
             errorMsg += " Error: Unknown (PSO is null, but no error object was provided by Metal)";
        }
        throw std::runtime_error(errorMsg);
    }
    std::cout << "  Compute pipeline state created successfully." << std::endl;

    // Return the pointer managing the PSO.
    return pPSO;
}


// Creates a sampler state for texture sampling in kernels
NS::SharedPtr<MTL::SamplerState> createSamplerState(NS::SharedPtr<MTL::Device> pDevice) {
     MTL::SamplerDescriptor* pRawDesc = MTL::SamplerDescriptor::alloc()->init();
     NS::SharedPtr<MTL::SamplerDescriptor> pDesc = NS::TransferPtr(pRawDesc);

     pDesc->setMinFilter(MTL::SamplerMinMagFilterLinear);
     pDesc->setMagFilter(MTL::SamplerMinMagFilterLinear);
     pDesc->setSAddressMode(MTL::SamplerAddressModeClampToEdge); // Horizontal wrap mode
     pDesc->setTAddressMode(MTL::SamplerAddressModeClampToEdge); // Vertical wrap mode
     pDesc->setNormalizedCoordinates(true); // Use coords 0.0-1.0

     MTL::SamplerState* pRawState = pDevice->newSamplerState(pDesc.get());
     NS::SharedPtr<MTL::SamplerState> pState = NS::TransferPtr(pRawState);
     if (!pState) {
         throw std::runtime_error("Failed to create sampler state.");
     }
     return pState;
}


// this function encodes commands to execute compute kernels and waits for completion
void runComputeKernel(NS::SharedPtr<MTL::Device> pDevice,
                      NS::SharedPtr<MTL::ComputePipelineState> pPSO,
                      NS::SharedPtr<MTL::Texture> pInputTexture,
                      NS::SharedPtr<MTL::Texture> pOutputTexture,
                      NS::SharedPtr<MTL::SamplerState> pSamplerState, // Can be null if not needed
                      const ConvolutionInfo* pConvInfo)               // Can be null if not needed
{
    if (!pDevice || !pPSO || !pInputTexture || !pOutputTexture) {
        throw std::invalid_argument("Invalid arguments provided for kernel execution.");
    }
    
    // Create Command Queue
    // this creates 1 queue. If we need parallel submissions, we can have more queues.
    // similar to OpenCL command queue.
    NS::SharedPtr<MTL::CommandQueue> pCommandQueue = NS::TransferPtr(pDevice->newCommandQueue());
    if (!pCommandQueue) { throw std::runtime_error("Failed to create Metal command queue."); }

    
    // Objective-C memory management
    // by creating a Autorelease Pool, it ensures objects like command buffers and command encoders
    // are cleaned up properly after kernel execution
    NS::AutoreleasePool* pLocalPool = NS::AutoreleasePool::alloc()->init(); // For temp objects like buffer/encoder
    if (!pLocalPool) { throw std::runtime_error("Failed to create local autorelease pool."); }

    
    MTL::CommandBuffer* pCmdBuffer = nullptr;
    MTL::CommandBufferStatus status = MTL::CommandBufferStatusNotEnqueued; // Track status
    NS::SharedPtr<NS::Error> pError;
    NS::SharedPtr<MTL::Buffer> pKernelMatrixBuffer; // Hold buffers within this scope
    NS::SharedPtr<MTL::Buffer> pKernelDimBuffer;
    
    try {
        // Create Command Buffer
        // Get a command buffer from the queue
        // its memory lifetime will be tied to pLocalPool.
        pCmdBuffer = pCommandQueue->commandBuffer();
        if (!pCmdBuffer) throw std::runtime_error("Failed to create command buffer from queue.");
        
        // Create Compute Command Encoder
        // Get an encoder specifically for compute commands
        MTL::ComputeCommandEncoder* pCmdEncoder = pCmdBuffer->computeCommandEncoder();
        if (!pCmdEncoder) throw std::runtime_error("Failed to create compute command encoder from buffer.");
        
        // Encode Commands
        // Set the pipeline state (which kernel to run)
        pCmdEncoder->setComputePipelineState(pPSO.get());
        
        // Bind resources (textures) to argument indices used in the shader.
        // setTexture(texture, index) maps the texture to `texture<float, access::read>` at texture(index) in the kernel
        pCmdEncoder->setTexture(pInputTexture.get(), 0);  // Bind input texture to index 0
        pCmdEncoder->setTexture(pOutputTexture.get(), 1); // Bind output texture to index 1
        
        // Set sampler state if provided (needed for k_convolve, k_edge_detect)
        if (pSamplerState) {
             pCmdEncoder->setSamplerState(pSamplerState.get(), 0); // Sampler at index 0
        }
        
        // Set convolution kernel buffer if provided (needed for k_convolve)
        if (pConvInfo) {
            if (pConvInfo->matrix.empty()) {
                 throw std::runtime_error("ConvolutionInfo provided but matrix is empty.");
            }
            // Create buffer for the matrix
             pKernelMatrixBuffer = NS::TransferPtr(pDevice->newBuffer(pConvInfo->matrix.data(),
                                                                      pConvInfo->matrix.size() * sizeof(float),
                                                                      MTL::ResourceStorageModeShared)); // Shared memory is efficient on UMA
             if (!pKernelMatrixBuffer) throw std::runtime_error("Failed to create buffer for kernel matrix.");

             // Create buffer for the dimension
             pKernelDimBuffer = NS::TransferPtr(pDevice->newBuffer(&pConvInfo->dimension,
                                                                    sizeof(int),
                                                                    MTL::ResourceStorageModeShared));
             if (!pKernelDimBuffer) throw std::runtime_error("Failed to create buffer for kernel dimension.");

             // Bind buffers to the kernel function arguments
             pCmdEncoder->setBuffer(pKernelMatrixBuffer.get(), 0, 0); // Matrix at buffer index 0
             pCmdEncoder->setBuffer(pKernelDimBuffer.get(), 0, 1);    // Dimension at buffer index 1
        }
        
        // Dispatch Threads (Kernel Launch Configuration)
        // (would be like <<<gridSize, blockSize>>> in CUDA)
        //    gridSize: Total number of threads to launch (usually matching output dimensions)
        //    threadgroupSize: Number of threads per group (workgroup/thread block)
        // Metal can query the PSO for optimal/valid sizes
        MTL::Size gridSize = MTL::Size::Make(pInputTexture->width(), pInputTexture->height(), 1); // Launch thread per pixel
        
        // Query the PSO for hardware-recommended threadgroup dimensions.
        NS::UInteger w = pPSO->threadExecutionWidth(); // Optimal width (Warp size)
        NS::UInteger h = pPSO->maxTotalThreadsPerThreadgroup() / w; // Calculate height based on max total size
        if (h == 0)
            h = 1; // Ensure height is at least 1
        MTL::Size threadgroupSize = MTL::Size::Make(w, h, 1); // Use recommended width, calculated height
        
        std::cout << "  Dispatching Metal kernel: Grid(" << gridSize.width << "x" << gridSize.height
        << "), Threadgroup(" << threadgroupSize.width << "x" << threadgroupSize.height << ")" << std::endl;
        
        // Encode the actual dispatch command.
        pCmdEncoder->dispatchThreads(gridSize, threadgroupSize);
        
        // Finalize this encoder. No more compute commands can be added.
        pCmdEncoder->endEncoding();
        
        // Submit all encoded commands to the command queue for the GPU to execute.
        pCmdBuffer->commit();
        std::cout << "  Kernel submitted to GPU. Waiting for completion..." << std::endl;
        
        // Wait on the CPU for the GPU to finish executing this command buffer.
        pCmdBuffer->waitUntilCompleted();
        std::cout << "  GPU execution finished." << std::endl;
        
        // Check for Errors During Execution (via its status)
        status = pCmdBuffer->status();
        if (status == MTL::CommandBufferStatusError) {
            std::cerr << "  Command buffer execution failed!" << std::endl;
            NS::Error* pRawCmdError = pCmdBuffer->error();
            if(pRawCmdError) { pError = NS::TransferPtr(pRawCmdError->retain()); } // Retain error
        }
    } catch (const std::exception& e) {
        std::cerr << "  Exception during kernel encoding/submission: " << e.what() << std::endl;
        pLocalPool->release(); throw; // Rethrow after releasing pool
    } catch (...) {
        std::cerr << "  Unknown exception during kernel encoding/submission." << std::endl;
        pLocalPool->release(); throw std::runtime_error("Unknown error during kernel execution.");
    }
    
    // release the pool. This destroys the pool object and sends a release message
    // to all objects added to it (like pCmdBuffer, pCmdEncoder, pRawCmdError, etc)
    pLocalPool->release();
    
    if (pError) {
        std::string errorMsg = "CommandBuffer execution failed. Error: " + std::string(pError->localizedDescription()->utf8String());
        throw std::runtime_error(errorMsg);
    } else if (status == MTL::CommandBufferStatusError) {
        throw std::runtime_error("CommandBuffer failed with unknown Metal error (Status Error, no details).");
    } else if (status != MTL::CommandBufferStatusCompleted) {
        throw std::runtime_error("CommandBuffer finished with unexpected status: " + std::to_string(status));
    }
    
}


// this function downloads pixel data from Metal Texture on GPU back onto the CPU buffer
// similar to using cudaMemcpyFromArray in CUDA
void downloadTextureData(NS::SharedPtr<MTL::Texture> pTexture,
                         std::vector<unsigned char>& outputImageData)
{
     if (!pTexture) {
        throw std::invalid_argument("Invalid texture provided for download.");
    }
    NS::UInteger width = pTexture->width();
    NS::UInteger height = pTexture->height();
    NS::UInteger bytesPerRow = width * 4; // Stride for RGBA data
    size_t expectedSize = static_cast<size_t>(width) * height * 4;

    // Ensure the output vector has the correct size.
    if (outputImageData.size() != expectedSize) {
        outputImageData.resize(expectedSize);
    }

    // Define the region of the texture to read from. in this case, the entire texture (x=0, y=0)
    MTL::Region region = MTL::Region::Make2D(0, 0, width, height);

    // getBytes copies data from the specified texture region to the CPU buffer.
    pTexture->getBytes(static_cast<void*>(outputImageData.data()), // Destination CPU buffer
                       bytesPerRow,  // Destination stride
                       region,       // Source region on GPU texture
                       0);           // Mipmap (not used)
}


// this function saves image data from CPU buffer to output file
void saveImageData(const std::string& outputPath,
                   int width,
                   int height,
                   const std::vector<unsigned char>& imageData)
{
    if (imageData.empty()) {
        throw std::invalid_argument("Image data is empty, cannot save.");
    }
    
    // check for file extension
    std::string ext = outputPath.substr(outputPath.find_last_of(".") + 1);
    int success = 0;
    int strideBytes = width * 4;

    // Call the appropriate stb_write function
    if (ext == "png" || ext == "PNG") {
        success = stbi_write_png(outputPath.c_str(), width, height, 4, imageData.data(), strideBytes);
    } else if (ext == "jpg" || ext == "JPG" || ext == "jpeg" || ext == "JPEG") {
        success = stbi_write_jpg(outputPath.c_str(), width, height, 4, imageData.data(), 90);
    } else if (ext == "bmp" || ext == "BMP") {
        success = stbi_write_bmp(outputPath.c_str(), width, height, 4, imageData.data());
    } else if (ext == "tga" || ext == "TGA") {
        success = stbi_write_tga(outputPath.c_str(), width, height, 4, imageData.data());
    } else {
        throw std::runtime_error("Unsupported output file extension: " + ext + ". Use png, jpg, bmp, or tga.");
    }
    if (!success) {
        throw std::runtime_error("Failed to write image file to: " + outputPath);
    }
}


// main function
int main(int argc, const char * argv[]) {

    std::cout << "Starting Metal Image Processor..." << std::endl;

    // --- Configuration (from arguments) ---
    std::string inputPath;     // Image to load
    std::string outputPath;   // Image to save
    std::string kernelSequenceStr; // Comma-separated kernel names from user
    std::vector<std::string> kernelSequence; // Parsed sequence of kernel names
    // Use RGBA8Unorm for better compatibility with standard image processing math (normalized 0-1)
    MTL::PixelFormat pixelFormat = MTL::PixelFormatRGBA8Unorm;
    int benchmarkIterations = 0;
    
    // configure cxxopts (for CLI arguments)
    cxxopts::Options options("MetalImageProcessor", "Applies Metal compute kernels to images.");
    options.add_options()
        ("i,input", "Input image file path", cxxopts::value<std::string>(inputPath))
        ("o,output", "Output image file path", cxxopts::value<std::string>(outputPath))
        ("k,kernels", "Comma-separated sequence of kernels (e.g., 'gaussian_blur_3x3,sharpen_3x3')",
            cxxopts::value<std::string>(kernelSequenceStr)->default_value("grayscale"))
        ("benchmark", "Number of iterations to benchmark Gaussian Blur 3x3 vs CPU", cxxopts::value<int>(benchmarkIterations)->default_value("0"))
        ("h,help", "Print usage information");

    std::vector<KernelInfo> selectedKernelInfos; // Store info for the sequence
    std::vector<std::string> metalKernelNames; // Store Metal function names for the sequence
    bool isBenchmarkMode = false;
    
    
    try {
        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            std::cout << "\nAvailable kernels for -k/--kernels:" << std::endl;
            for(const auto& pair : KERNEL_REGISTRY) { std::cout << "  - " << pair.first << std::endl; }
            std::cout << "\nBenchmark mode (--benchmark N) always uses '" << BENCHMARK_KERNEL_USER_NAME << "'." << std::endl;
            return 0;
        }

        if (!result.count("input")) throw std::runtime_error("Missing --input");
        if (!result.count("output")) throw std::runtime_error("Missing --output");
        if (!std::filesystem::exists(inputPath)) throw std::runtime_error("Input file not found: " + inputPath);

        if (benchmarkIterations > 0) {
            // Benchmark mode: Use N iterations of the hardcoded benchmark kernel
            if (KERNEL_REGISTRY.find(BENCHMARK_KERNEL_USER_NAME) == KERNEL_REGISTRY.end()) {
                 throw std::runtime_error("Internal error: Benchmark kernel '" + BENCHMARK_KERNEL_USER_NAME + "' not found in registry.");
            }
            for (int i = 0; i < benchmarkIterations; ++i) {
                kernelSequence.push_back(BENCHMARK_KERNEL_USER_NAME);
            }
            isBenchmarkMode = true;
            std::cout << "Benchmark Mode: Applying '" << BENCHMARK_KERNEL_USER_NAME << "' " << benchmarkIterations << " times." << std::endl;
        } else {
            // Normal mode: parse kernelSequenceStr from -k/--kernels
            kernelSequence = split(kernelSequenceStr, ',');
            if (kernelSequence.empty()) {
                throw std::runtime_error("Kernel sequence is empty via -k/--kernels.");
            }
        }

        std::cout << "Starting Metal Image Processor..." << std::endl;
        std::cout << "  Input:  " << inputPath << std::endl;
        std::cout << "  Output: " << outputPath << std::endl;
        if (!isBenchmarkMode) std::cout << "  Kernel Sequence:" << std::endl;

        // --- Prepare kernel info for the sequence (either benchmark or user-defined) ---
        for (const auto& userKernelName : kernelSequence) {
            if (KERNEL_REGISTRY.find(userKernelName) == KERNEL_REGISTRY.end()) {
                throw std::runtime_error("Unknown kernel '" + userKernelName + "' in sequence.");
            }
            KernelInfo info = KERNEL_REGISTRY[userKernelName];
            selectedKernelInfos.push_back(info);
            std::string metalName = std::visit([](auto&& arg) -> std::string {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, SimpleKernelInfo>) return arg.metalKernelName;
                else if constexpr (std::is_same_v<T, ConvolutionInfo>) return arg.metalKernelName;
                return "";
            }, info);
            if (metalName.empty()) throw std::runtime_error("Internal error: Metal kernel name missing for " + userKernelName);
            metalKernelNames.push_back(metalName);
            if (!isBenchmarkMode) std::cout << "    - " << userKernelName << " (using Metal function: " << metalName << ")" << std::endl;
        }

    } catch (const cxxopts::exceptions::exception& e) {
        std::cerr << "Error parsing options: " << e.what() << std::endl; return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl; return 1;
    }
    


    // Memory Management
    // Declare Metal objects using NS::SharedPtr, so their lifetime will be automatically managed
    NS::SharedPtr<MTL::Device> pDevice;
    NS::SharedPtr<MTL::Texture> pTextureA; // For ping-ponging
    NS::SharedPtr<MTL::Texture> pTextureB; // For ping-ponging
    NS::SharedPtr<MTL::SamplerState> pSamplerState; // Create once if any kernel needs it
    // Map to store Pipeline State Objects, keyed by Metal kernel name to avoid recreation
    std::map<std::string, NS::SharedPtr<MTL::ComputePipelineState>> psoCache;
    
    std::vector<unsigned char> cpuImageData, cpuOutputImageDataGpu, cpuOutputImageDataCpu;
    int width = 0, height = 0;

    try {
        // 1. Initialize Metal Device (GPU)
        pDevice = setupMetalDevice();

        // 2. Load image from file into a vector on CPU memory
        loadImageData(inputPath, width, height, cpuImageData);
        std::cout << "Loaded input image '" << inputPath << "' (" << width << "x" << height << ")." << std::endl;

        // 3. Create Metal Textures on the GPU
        // Input texture usage depends on kernel type (read for grayscale, sample for convolution)
        MTL::TextureUsage textureUsage = MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite;
        bool needsSampling = false;
        for (const auto& info : selectedKernelInfos) {
             if (std::holds_alternative<ConvolutionInfo>(info) ||
                 (std::holds_alternative<SimpleKernelInfo>(info) && std::get<SimpleKernelInfo>(info).metalKernelName == "k_edge_detect"))
             {
                 needsSampling = true;
                 // Sampling requires ShaderRead usage, which is already included.
                 break;
             }
        }
        
        pTextureA = createMetalTexture(pDevice, width, height, pixelFormat, textureUsage);
        pTextureB = createMetalTexture(pDevice, width, height, pixelFormat, textureUsage);
        std::cout << "Created Metal processing textures A and B on device." << std::endl;
        
        // start benchmark timing
        auto gpu_start_time = std::chrono::high_resolution_clock::now();

        // 4. Upload CPU image data to the input GPU texture
        uploadImageDataToTexture(cpuImageData, pTextureA);
        std::cout << "Uploaded initial image data to texture A." << std::endl;

        // 5. Create Sampler State (only if needed)
        if (needsSampling) {
             pSamplerState = createSamplerState(pDevice);
             std::cout << "Created sampler state." << std::endl;
        }
        
        // 6. Process the kernel sequence
        NS::SharedPtr<MTL::Texture> pCurrentInputTexture = pTextureA;
        NS::SharedPtr<MTL::Texture> pCurrentOutputTexture = pTextureB;
        
        for (size_t i = 0; i < kernelSequence.size(); ++i) {
            const std::string& metalKernelName = metalKernelNames[i];
            const KernelInfo& kernelInfo = selectedKernelInfos[i];
            if (!isBenchmarkMode) std::cout << "\nApplying GPU filter " << (i + 1) << "/" << kernelSequence.size() << ": '" << kernelSequence[i] << "'..." << std::endl;



            // 6a. Get or create the Pipeline State Object (PSO)
            NS::SharedPtr<MTL::ComputePipelineState> pCurrentPSO;
            if (psoCache.count(metalKernelName)) {
                pCurrentPSO = psoCache[metalKernelName];
                std::cout << "  Using cached PSO for '" << metalKernelName << "'." << std::endl;
            } else {
                pCurrentPSO = setupPipelineState(pDevice, metalKernelName);
                psoCache[metalKernelName] = pCurrentPSO; // Cache it
                std::cout << "  Created and cached PSO for '" << metalKernelName << "'." << std::endl;
            }

            // 6b. Get pointer to ConvolutionInfo if needed
            const ConvolutionInfo* pConvInfoPtr = std::get_if<ConvolutionInfo>(&kernelInfo);

            // 6c. Run the kernel
            runComputeKernel(pDevice, pCurrentPSO,
                             pCurrentInputTexture, pCurrentOutputTexture, // Pass current input/output
                             pSamplerState,        // Pass sampler (null if not created)
                             pConvInfoPtr);        // Pass conv info (null if not needed)

            // 6d. Swap textures for the next iteration (Ping-Pong)
            std::swap(pCurrentInputTexture, pCurrentOutputTexture);
        }
        cpuOutputImageDataGpu.resize(static_cast<size_t>(width) * height * 4);
        std::cout << "\nExecution of filter sequence complete." << std::endl;

        
        // 7. Download processed data from the output GPU texture back to CPU memory
        cpuOutputImageDataGpu.resize(static_cast<size_t>(width) * height * 4);
        downloadTextureData(pCurrentInputTexture, cpuOutputImageDataGpu); // Download from the last input texture
        std::cout << "Downloaded processed data." << std::endl;
        
        // end benchmark timing
        auto gpu_end_time = std::chrono::high_resolution_clock::now();
        auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end_time - gpu_start_time);


        // 8. Save the processed CPU image data to a file
        saveImageData(outputPath, width, height, cpuOutputImageDataGpu);
        std::cout << "Saved output image to: '" << outputPath << "'." << std::endl;

        // --- CPU Benchmarking (if flag enabled) ---
        if (isBenchmarkMode) {
            std::cout << "\n--- Benchmark Results for " << benchmarkIterations << " applications of '" << BENCHMARK_KERNEL_USER_NAME << "' ---" << std::endl;
            std::cout << "GPU Total Time (upload + " << benchmarkIterations << " kernels + download): " << gpu_duration.count() << " ms" << std::endl;

            // CPU Timing
            std::cout << "Running CPU benchmark..." << std::endl;
            cpuOutputImageDataCpu = cpuImageData; // Start with a fresh copy

            auto cpu_start_time = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < benchmarkIterations; ++i) {
                // cpuGaussianBlur3x3 modifies in place. For N iterations, apply N times sequentially.
                 if (i == 0) {
                    cpuOutputImageDataCpu = cpuImageData; // Use original for first pass
                    cpuGaussianBlur3x3(cpuOutputImageDataCpu, width, height);
                 } else {
                    // cpuOutputImageDataCpu holds result from iteration i-1
                    cpuGaussianBlur3x3(cpuOutputImageDataCpu, width, height);
                 }
            }
            auto cpu_end_time = std::chrono::high_resolution_clock::now();
            auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end_time - cpu_start_time);

            std::cout << "CPU Total Time (" << benchmarkIterations << " kernels): " << cpu_duration.count() << " ms" << std::endl;

            // Comparison
            if (benchmarkIterations > 0) {
                double gpu_avg = static_cast<double>(gpu_duration.count()) / benchmarkIterations; // Rough avg including I/O split
                double cpu_avg = static_cast<double>(cpu_duration.count()) / benchmarkIterations;
                std::cout << "Approx Average Time per iteration: GPU ~= " << gpu_avg << " ms, CPU = " << cpu_avg << " ms" << std::endl;
                if (gpu_duration.count() > 0 && cpu_duration.count() > 0) {
                     std::cout << "CPU / GPU (Total Time Ratio): " << static_cast<double>(cpu_duration.count()) / gpu_duration.count() << std::endl;
                }
            }
            saveImageData("output_cpu_benchmark.png", width, height, cpuOutputImageDataCpu);
        } else {
             std::cout << "Processing complete." << std::endl;
        }

    } catch (const std::exception& e) {
        // Catch standard C++ exceptions
        std::cerr << "\n Error: " << e.what() << std::endl;
        return 1; // failure
    } catch (...) {
        // Catch non-standard exceptions
        std::cerr << "\n Unknown Error occurred." << std::endl;
        return 1; // failure
    }

    std::cout << "Execution finished successfully." << std::endl;
    return 0; // success
}
