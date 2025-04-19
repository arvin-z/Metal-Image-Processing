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


// this function encodes commands to execute the kernel and waits for completion
void runSampleKernel(NS::SharedPtr<MTL::Device> pDevice, // Accepts SharedPtrs
                          NS::SharedPtr<MTL::ComputePipelineState> pPSO,
                          NS::SharedPtr<MTL::Texture> pInputTexture,
                          NS::SharedPtr<MTL::Texture> pOutputTexture)
{
    if (!pDevice || !pPSO || !pInputTexture || !pOutputTexture) {
        throw std::invalid_argument("Invalid arguments provided for kernel execution.");
    }
    
    // Create Command Queue
    // this creates 1 queue. If we need parallel submissions, we can have more queues.
    // similar to OpenCL command queue.
    MTL::CommandQueue* pRawQueue = pDevice->newCommandQueue();
    NS::SharedPtr<MTL::CommandQueue> pCommandQueue = NS::TransferPtr(pRawQueue);
    if (!pCommandQueue) {
        throw std::runtime_error("Failed to create Metal command queue.");
    }
    
    // Objective-C memory management
    // by creating a Autorelease Pool, it ensures objects like command buffers and command encoders
    // are cleaned up properly after kernel execution
    NS::AutoreleasePool* pLocalPool = NS::AutoreleasePool::alloc()->init();
    if (!pLocalPool) {
        throw std::runtime_error("Failed to create local autorelease pool for command buffer/encoder.");
    }
    
    MTL::CommandBuffer* pCmdBuffer = nullptr;
    MTL::CommandBufferStatus status = MTL::CommandBufferStatusNotEnqueued; // Track status
    NS::SharedPtr<NS::Error> pError;
    
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
            if(pRawCmdError) {
                // Retain the error object, so pool doesn't releases it
                pError = NS::TransferPtr(pRawCmdError->retain());
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "  Exception during kernel encoding/submission: " << e.what() << std::endl;
        pLocalPool->release();
        throw;
    } catch (...) {
        std::cerr << "  Unknown exception during kernel encoding/submission." << std::endl;
        pLocalPool->release();
        throw std::runtime_error("Unknown error occurred during kernel execution.");
    }
    
    // release the pool. This destroys the pool object and sends a release message
    // to all objects added to it (like pCmdBuffer, pCmdEncoder, pRawCmdError, etc)
    pLocalPool->release();
    
    if (pError) {
        std::string errorMsg = "CommandBuffer execution failed.";
        errorMsg += " Error: " + std::string(pError->localizedDescription()->utf8String());
        throw std::runtime_error(errorMsg);
    } else if (status == MTL::CommandBufferStatusError) {
        throw std::runtime_error("CommandBuffer execution failed with an unknown Metal error (Status is Error, no details captured)");
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

    std::cout << "Starting preliminary Metal Image Processor..." << std::endl;

    // --- Configuration ---
    std::string inputPath = "input.png";     // Image to load
    std::string outputPath = "output.png";   // Image to save
    std::string kernelName = "k_grayscale";  // Name of kernel function in .metal file
    MTL::PixelFormat pixelFormat = MTL::PixelFormatRGBA8Unorm_sRGB; // sRGB pixel format


    // Memory Management
    // Declare Metal objects using NS::SharedPtr, so their lifetime will be automatically managed
    NS::SharedPtr<MTL::Device> pDevice;
    NS::SharedPtr<MTL::Texture> pInputTexture;
    NS::SharedPtr<MTL::Texture> pOutputTexture;
    NS::SharedPtr<MTL::ComputePipelineState> pPSO;

    try {
        // 1. Initialize Metal Device (GPU)
        pDevice = setupMetalDevice();

        // 2. Load image from file into a vector on CPU memory
        int width = 0, height = 0;
        std::vector<unsigned char> cpuImageData;
        loadImageData(inputPath, width, height, cpuImageData);
        std::cout << "Loaded input image '" << inputPath << "' (" << width << "x" << height << ")." << std::endl;

        // 3. Create Metal Textures on the GPU
        // Input texture: Shader will read from it
        pInputTexture = createMetalTexture(pDevice, width, height, pixelFormat, MTL::TextureUsageShaderRead);
        // Output texture: Shader will write to it
        pOutputTexture = createMetalTexture(pDevice, width, height, pixelFormat, MTL::TextureUsageShaderWrite);
        std::cout << "Created Metal input and output textures on device." << std::endl;

        // 4. Upload CPU image data to the input GPU texture
        uploadImageDataToTexture(cpuImageData, pInputTexture);
        std::cout << "Uploaded image data to input texture." << std::endl;


        // 5. Load and compile the Metal kernel into a Pipeline State Object (PSO)
        pPSO = setupPipelineState(pDevice, kernelName);
        std::cout << "Setup compute pipeline state for kernel: '" << kernelName << "'." << std::endl;

        
        // 6. Run the kernel: Encode commands, submit, wait for completion.
        runSampleKernel(pDevice, pPSO, pInputTexture, pOutputTexture);
        std::cout << "Metal kernel execution complete." << std::endl;

        
        // 7. Download processed data from the output GPU texture back to CPU memory
        std::vector<unsigned char> cpuOutputImageData;
        cpuOutputImageData.resize(static_cast<size_t>(width) * height * 4);
        downloadTextureData(pOutputTexture, cpuOutputImageData);
        std::cout << "Downloaded processed data from output texture." << std::endl;

        // 8. Save the processed CPU image data to a file
        saveImageData(outputPath, width, height, cpuOutputImageData);
        std::cout << "Saved output image to: '" << outputPath << "'." << std::endl;

        std::cout << "Processing complete..." << std::endl;

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
