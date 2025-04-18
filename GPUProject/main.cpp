//
//  main.cpp
//  GPUProject
//
//  Main program demonstrating Metal image processing (Corrected Refactor)
//

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <memory>
#include <thread>        // For potential sleep (debugging)
#include <chrono>        // For potential sleep (debugging)


// stb (for image file loading and writing)
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

// Metal framework implementation includes
// These macros are required before including Metal headers in 1 cpp file
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>


/**
 * @brief Creates the default Metal device and manages it with NS::SharedPtr.
 * (Unchanged from your fixed version)
 */
NS::SharedPtr<MTL::Device> setupMetalDevice() {
    MTL::Device* pRawDevice = MTL::CreateSystemDefaultDevice();
    if (!pRawDevice) {
        throw std::runtime_error("Failed to get default Metal device.");
    }

    // Use TransferPtr to manage the device lifecycle with smart pointer
    NS::SharedPtr<MTL::Device> pDevice = NS::TransferPtr(pRawDevice);
    std::cout << "Using Metal Device: " <<
        (pDevice->name() ? pDevice->name()->utf8String() : "Unknown") << std::endl;

    return pDevice;
}


// uses the C stb library to load an image file into memory. (Unchanged)
void loadImageData(const std::string& inputPath,
                   int& width,
                   int& height,
                   std::vector<unsigned char>& imageData)
{
    int channels = 0;
    unsigned char* loadedPixels = stbi_load(inputPath.c_str(), &width, &height, &channels, 4);
    if (!loadedPixels) {
        throw std::runtime_error("Failed to load image file: " + inputPath + " - Reason: " + stbi_failure_reason());
    }
    size_t imageSize = static_cast<size_t>(width) * height * 4;
    imageData.resize(imageSize);
    std::copy(loadedPixels, loadedPixels + imageSize, imageData.begin());
    stbi_image_free(loadedPixels);
}


/**
 * @brief Creates a Metal texture using metal-cpp smart pointers.
 * (Accepts SharedPtr<Device> as in your fixed version)
 */
NS::SharedPtr<MTL::Texture> createMetalTexture(NS::SharedPtr<MTL::Device> pDevice, // Accepts SharedPtr
                                               int width,
                                               int height,
                                               MTL::PixelFormat format,
                                               MTL::TextureUsage usage,
                                               MTL::StorageMode storage = MTL::StorageModeShared)
{
    // Use .get() if needed for raw pointer, but descriptor creation takes format etc directly
    NS::SharedPtr<MTL::TextureDescriptor> pTextureDesc = NS::TransferPtr(MTL::TextureDescriptor::texture2DDescriptor(format, width, height, false));
    if (!pTextureDesc) { throw std::runtime_error("Failed to create Texture Descriptor."); }

    pTextureDesc->setUsage(usage);
    pTextureDesc->setStorageMode(storage);

    // Pass raw device pointer via .get()
    NS::SharedPtr<MTL::Texture> pTexture = NS::TransferPtr(pDevice->newTexture(pTextureDesc.get()));
    if (!pTexture) { throw std::runtime_error("Failed to create texture."); }

    return pTexture;
}


// Uploads image data to texture. (Unchanged)
void uploadImageDataToTexture(const std::vector<unsigned char>& imageData,
                              NS::SharedPtr<MTL::Texture> pTexture) // Accepts SharedPtr
{
    if (!pTexture || imageData.empty()) {
        throw std::invalid_argument("Invalid texture or image data for upload.");
    }
    NS::UInteger width = pTexture->width();
    NS::UInteger height = pTexture->height();
    NS::UInteger bytesPerRow = width * 4;
    MTL::Region region = MTL::Region::Make2D(0, 0, width, height);
    pTexture->replaceRegion(region, 0, 0, static_cast<const void*>(imageData.data()), bytesPerRow, 0);
}


/**
 * @brief Sets up the compute pipeline state, correctly handling potential errors.
 * (Error handling fixed, manual string release removed)
 */
NS::SharedPtr<MTL::ComputePipelineState> setupPipelineState(NS::SharedPtr<MTL::Device> pDevice, // Accepts SharedPtr
                                                            const std::string& kernelName)
{
    NS::Error* pRawError = nullptr; // Raw pointer for potential error object

    // --- Load default library ---
    MTL::Library* pRawLibrary = pDevice->newDefaultLibrary(); // Assumed +1
    NS::SharedPtr<MTL::Library> pDefaultLibrary = NS::TransferPtr(pRawLibrary);
    if (!pDefaultLibrary) {
        throw std::runtime_error("Failed to load default Metal library.");
    }
    std::cout << "  Loaded default Metal library." << std::endl;

    // --- Load kernel function ---
    // NS::String::string creates an autoreleased string, okay for transient use.
    // No need to manually release it.
    NS::String* pKernelNsString = NS::String::string(kernelName.c_str(), NS::UTF8StringEncoding);
    MTL::Function* pRawFunction = pDefaultLibrary->newFunction(pKernelNsString); // Assumed +1
    NS::SharedPtr<MTL::Function> pFunction = NS::TransferPtr(pRawFunction);
    // **** REMOVED incorrect: pKernelNsString->release(); ****

    if (!pFunction) {
        throw std::runtime_error("Failed to find kernel function: " + kernelName);
    }
     std::cout << "  Loaded kernel function: " << kernelName << std::endl;

    // --- Create compute pipeline state ---
     std::cout << "  Creating compute pipeline state..." << std::endl;
    MTL::ComputePipelineState* pRawPSO = pDevice->newComputePipelineState(pFunction.get(), &pRawError); // Returns +1 PSO, maybe autoreleased Error
    NS::SharedPtr<MTL::ComputePipelineState> pPSO = NS::TransferPtr(pRawPSO); // Manages PSO

    // --- Correct Error Handling ---
    NS::SharedPtr<NS::Error> pErrorWrapper; // Smart pointer for the error
    if (pRawError) {
        // If an error object exists (pRawError is not nullptr), it's likely autoreleased.
        // Retain it explicitly before transferring ownership to the SharedPtr.
        std::cout << "  Capturing PSO creation error..." << std::endl;
        pErrorWrapper = NS::TransferPtr(pRawError->retain());
    }
    // -----------------------------

    if (!pPSO) { // Check if PSO creation failed (pRawPSO would be null)
        std::string errorMsg = "Failed to create compute pipeline state for " + kernelName + ".";
        if (pErrorWrapper) { // Check the smart pointer holding the retained error
            errorMsg += " Error: " + std::string(pErrorWrapper->localizedDescription()->utf8String());
        } else if (pRawError) {
             errorMsg += " Error: Code " + std::to_string(pRawError->code()) + " (Error object not retained?)";
        } else {
             errorMsg += " Error: Unknown (No PSO and no error object returned).";
        }
        throw std::runtime_error(errorMsg);
    }
    std::cout << "  Compute pipeline state created." << std::endl;

    // pDefaultLibrary, pFunction, pErrorWrapper (if created) are managed by SharedPtr
    return pPSO; // Returns the managed PSO
}


/**
 * @brief Runs the compute kernel, using a local autorelease pool.
 * (Local pool restored, error handling uses retain+TransferPtr)
 */
void runSampleKernel(NS::SharedPtr<MTL::Device> pDevice, // Accepts SharedPtr
                          NS::SharedPtr<MTL::ComputePipelineState> pPSO,
                          NS::SharedPtr<MTL::Texture> pInputTexture,
                          NS::SharedPtr<MTL::Texture> pOutputTexture)
{
    if (!pDevice || !pPSO || !pInputTexture || !pOutputTexture) {
        throw std::invalid_argument("Invalid arguments for running kernel.");
    }

    // Create Command Queue using SharedPtr
    NS::SharedPtr<MTL::CommandQueue> pCommandQueue = NS::TransferPtr(pDevice->newCommandQueue());
    if (!pCommandQueue) {
        throw std::runtime_error("Failed to create command queue.");
    }

    // ** RESTORED Local Autorelease Pool for Command Buffer/Encoder **
    NS::AutoreleasePool* pLocalPool = NS::AutoreleasePool::alloc()->init();
    if (!pLocalPool) {
         throw std::runtime_error("Failed to create local autorelease pool in runSampleKernel.");
    }
    std::cout << "  Local pool created for kernel execution." << std::endl;

    MTL::CommandBuffer* pCmdBuffer = nullptr; // Declare outside try
    MTL::CommandBufferStatus status = MTL::CommandBufferStatusNotEnqueued; // Init status
    NS::SharedPtr<NS::Error> pError; // Use SharedPtr to manage potential error object

    try {
        // Command Buffer and Encoder are likely autoreleased, managed by pLocalPool
        pCmdBuffer = pCommandQueue->commandBuffer();
        if (!pCmdBuffer) throw std::runtime_error("Failed to create command buffer.");
        MTL::ComputeCommandEncoder* pCmdEncoder = pCmdBuffer->computeCommandEncoder();
        if (!pCmdEncoder) throw std::runtime_error("Failed to create compute command encoder.");

        pCmdEncoder->setComputePipelineState(pPSO.get());
        pCmdEncoder->setTexture(pInputTexture.get(), 0);
        pCmdEncoder->setTexture(pOutputTexture.get(), 1);

        // Calculate grid and threadgroup sizes
        MTL::Size gridSize = MTL::Size::Make(pInputTexture->width(), pInputTexture->height(), 1);
        NS::UInteger w = pPSO->threadExecutionWidth();
        NS::UInteger h = pPSO->maxTotalThreadsPerThreadgroup() / w;
        if (h == 0) h = 1;
        MTL::Size threadgroupSize = MTL::Size::Make(w, h, 1);
        std::cout << "Dispatching " << gridSize.width << "x" << gridSize.height
                  << " threads with group size " << threadgroupSize.width << "x" << threadgroupSize.height << std::endl;
        pCmdEncoder->dispatchThreads(gridSize, threadgroupSize);

        pCmdEncoder->endEncoding();
        pCmdBuffer->commit();
        std::cout << "  Kernel submitted. Waiting for completion..." << std::endl;
        pCmdBuffer->waitUntilCompleted();
        std::cout << "  Kernel execution completed." << std::endl;

        // --- CHECK STATUS *BEFORE* RELEASING POOL ---
        status = pCmdBuffer->status(); // Get status now
        if (status == MTL::CommandBufferStatusError) {
            // Get the error object (likely autoreleased), retain before transferring.
            std::cerr << "  Command buffer error detected!" << std::endl;
             NS::Error* pRawError = pCmdBuffer->error(); // Get raw potentially autoreleased error
             if(pRawError) {
                 pError = NS::TransferPtr(pRawError->retain()); // Retain + Transfer ownership
             }
        }
        // ---------------------------------------------

    } catch (const std::exception& e) {
        pLocalPool->release(); // Release pool even if exception occurs mid-try
        throw; // Re-throw the exception
    } catch (...) {
        pLocalPool->release(); // Release pool for non-std exceptions
        throw std::runtime_error("Unknown error occurred during kernel execution.");
    }

    // Release the local pool, cleaning up pCmdBuffer, pCmdEncoder, etc.
    pLocalPool->release();
    std::cout << "  Local autorelease pool drained." << std::endl;

    // Check the final status *after* the pool is drained
    // (pError holds the retained error object if needed)
    if (pError) { // Check if the error smart pointer was populated
         std::string errorMsg = "CommandBuffer execution failed.";
         errorMsg += " Error: " + std::string(pError->localizedDescription()->utf8String());
         throw std::runtime_error(errorMsg);
     } else if (status == MTL::CommandBufferStatusError) {
         // Fallback if status is error but we couldn't get/retain the error object
         throw std::runtime_error("CommandBuffer execution failed with an unknown error (status is Error).");
     } else if (status != MTL::CommandBufferStatusCompleted) {
          throw std::runtime_error("CommandBuffer did not complete successfully. Status: " + std::to_string(status));
     }
}


// Downloads texture data to CPU memory. (Unchanged)
void downloadTextureData(NS::SharedPtr<MTL::Texture> pTexture, // Accepts SharedPtr
                         std::vector<unsigned char>& outputImageData)
{
     if (!pTexture || outputImageData.empty()) {
        throw std::invalid_argument("Invalid texture or output buffer for download.");
    }
    NS::UInteger width = pTexture->width();
    NS::UInteger height = pTexture->height();
    NS::UInteger bytesPerRow = width * 4;
    size_t expectedSize = static_cast<size_t>(width) * height * 4;
    if (outputImageData.size() != expectedSize) {
        outputImageData.resize(expectedSize);
    }
    MTL::Region region = MTL::Region::Make2D(0, 0, width, height);
    pTexture->getBytes(static_cast<void*>(outputImageData.data()), bytesPerRow, region, 0);
}


// Saves image data to file using stb_image_write. (Unchanged)
void saveImageData(const std::string& outputPath,
                   int width,
                   int height,
                   const std::vector<unsigned char>& imageData)
{
    if (imageData.empty()) {
        throw std::invalid_argument("Image data is empty, cannot save.");
    }
    std::string ext = outputPath.substr(outputPath.find_last_of(".") + 1);
    int success = 0;
    int strideBytes = width * 4;

    if (ext == "png" || ext == "PNG") {
        success = stbi_write_png(outputPath.c_str(), width, height, 4, imageData.data(), strideBytes);
    } else if (ext == "jpg" || ext == "JPG" || ext == "jpeg" || ext == "JPEG") {
        success = stbi_write_jpg(outputPath.c_str(), width, height, 4, imageData.data(), 90);
    } else if (ext == "bmp" || ext == "BMP") {
        success = stbi_write_bmp(outputPath.c_str(), width, height, 4, imageData.data());
    } else if (ext == "tga" || ext == "TGA") {
        success = stbi_write_tga(outputPath.c_str(), width, height, 4, imageData.data());
    } else {
        throw std::runtime_error("Unsupported output file extension: " + ext);
    }
    if (!success) {
        throw std::runtime_error("Failed to write image file to: " + outputPath);
    }
}


int main(int argc, const char * argv[]) {

    // ** NO Top-Level NSAutoreleasePool **
    std::cout << "Starting GPU Project (Corrected Refactor)..." << std::endl;

    // --- Configuration ---
    std::string inputPath = "input.png";
    std::string outputPath = "output.png"; // Adjusted path for general use
    std::string kernelName = "k_grayscale";
    MTL::PixelFormat pixelFormat = MTL::PixelFormatRGBA8Unorm_sRGB;
    // ---------------------

    NS::SharedPtr<MTL::Device> pDevice;
    NS::SharedPtr<MTL::Texture> pInputTexture;
    NS::SharedPtr<MTL::Texture> pOutputTexture;
    NS::SharedPtr<MTL::ComputePipelineState> pPSO;

    try {
        pDevice = setupMetalDevice();

        int width = 0, height = 0;
        std::vector<unsigned char> cpuImageData;
        loadImageData(inputPath, width, height, cpuImageData);
        std::cout << "Loaded image: " << width << "x" << height << std::endl;

        // Pass SharedPtr directly to functions that now accept it
        pInputTexture = createMetalTexture(pDevice, width, height, pixelFormat, MTL::TextureUsageShaderRead);
        pOutputTexture = createMetalTexture(pDevice, width, height, pixelFormat, MTL::TextureUsageShaderWrite);
        std::cout << "Created Metal textures." << std::endl;

        uploadImageDataToTexture(cpuImageData, pInputTexture);
        std::cout << "Uploaded image data to input texture." << std::endl;

        pPSO = setupPipelineState(pDevice, kernelName);
        std::cout << "Setup compute pipeline state for kernel: " << kernelName << std::endl;

        runSampleKernel(pDevice, pPSO, pInputTexture, pOutputTexture);
        std::cout << "Executed Metal kernel." << std::endl;

        std::vector<unsigned char> cpuOutputImageData(static_cast<size_t>(width) * height * 4);
        downloadTextureData(pOutputTexture, cpuOutputImageData);
        std::cout << "Downloaded processed data from output texture." << std::endl;

        saveImageData(outputPath, width, height, cpuOutputImageData);
        std::cout << "Saved output image to: " << outputPath << std::endl;

        std::cout << "Main operations complete. Resources will be automatically released." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        // Smart pointers automatically release resources.
        std::cerr << "Exiting due to error." << std::endl;
        return 1;
    }

    // ** NO POOL RELEASE NEEDED HERE **
    std::cout << "Execution finished successfully." << std::endl;
    // All SharedPtrs (pDevice, pInputTexture, pOutputTexture, pPSO) go out of scope here
    // and release their underlying Metal objects automatically.
    return 0;
}
