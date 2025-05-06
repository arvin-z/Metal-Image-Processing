//
//  ProjectKernels.metal
//  GPUProject
//
//  Created by Arvin Ziaei on 2025-04-16.
//

#include <metal_stdlib>
using namespace metal;


// --- Sampler Definition ---
// Used by convolution kernels to read neighboring pixels.
// - coord::normalized: Use texture coordinates from 0.0 to 1.0.
// - address::clamp_to_edge: Pixels outside the texture return the color of the nearest edge pixel.
// - filter::linear: Interpolate between pixels if sampling between them
constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);

// Basic kernel: grayscale converter
kernel void k_grayscale(texture2d<half, access::read> inTexture [[texture(0)]],
                      texture2d<half, access::write> outTexture [[texture(1)]],
                      uint2 gid [[thread_position_in_grid]])
{
    // Check bounds
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) {
        return;
    }
    
    // Read color from input texture
    half4 color = inTexture.read(gid);
    
    // Convert to grayscale using luminance formula
    half luminance = 0.299h * color.r + 0.587h * color.g + 0.114h * color.b;
    
    // Create grayscale color (preserve alpha)
    half4 grayscaleColor = half4(luminance, luminance, luminance, color.a);
    
    // Write grayscale color to output texture
    outTexture.write(grayscaleColor, gid);
}

// Convolutional kernels
// base version (apply matrix via buffer) to input texture
kernel void k_convolve(texture2d<float, access::sample> inTexture [[texture(0)]], // Sample allows using sampler
                       texture2d<float, access::write> outTexture [[texture(1)]],
                       sampler imageSampler [[sampler(0)]],                       // Sampler state from C++
                       constant float* kernelMatrix [[buffer(0)]],                // Convolution matrix (e.g., 3x3 = 9 floats)
                       constant int& kernelDim [[buffer(1)]],                     // Dimension (e.g., 3 for 3x3)
                       uint2 gid [[thread_position_in_grid]])
{
    // Check bounds
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) {
        return;
    }

    // Calculate texture coordinates (0.0 - 1.0 range) for the center pixel
    float2 uv = float2(gid) / float2(inTexture.get_width(), inTexture.get_height());
    // Calculate the size of one pixel in texture coordinates
    float2 pixelSize = 1.0f / float2(inTexture.get_width(), inTexture.get_height());

    // Calculate the offset to the top-left corner of the kernel neighborhood
    int kernelRadius = kernelDim / 2; // e.g., 3/2 = 1

    // Accumulator for the weighted sum of neighboring pixels
    float4 resultColor = float4(0.0f);

    // Iterate over the kernel matrix (e.g., 3x3)
    for (int y = 0; y < kernelDim; ++y) {
        for (int x = 0; x < kernelDim; ++x) {
            // Calculate the offset from the center pixel for this neighbor
            int offsetX = x - kernelRadius; // e.g., 0-1 = -1, 1-1 = 0, 2-1 = 1
            int offsetY = y - kernelRadius;

            // Calculate the texture coordinate of the neighbor pixel
            float2 neighborUV = uv + float2(offsetX, offsetY) * pixelSize;

            // Read the neighbor pixel color using the sampler
            // Sampler handles boundary conditions (clamp_to_edge)
            float4 neighborColor = inTexture.sample(imageSampler, neighborUV);

            // Get the corresponding weight from the kernel matrix
            float weight = kernelMatrix[y * kernelDim + x];

            // Accumulate the weighted color (excluding alpha for convolution)
            resultColor.rgb += neighborColor.rgb * weight;
        }
    }

    // Read the original center pixel to preserve its alpha value
    float4 centerColor = inTexture.read(gid);
    resultColor.a = centerColor.a; // Preserve original alpha

    // Clamp color values to 0-1 range to avoid issues after convolution
    resultColor = saturate(resultColor);

    // Write the final convoluted color
    outTexture.write(resultColor, gid);
}


// --- Sobel Edge Detection Kernel ---
kernel void k_edge_detect(texture2d<float, access::sample> inTexture [[texture(0)]],
                          texture2d<float, access::write> outTexture [[texture(1)]],
                          sampler imageSampler [[sampler(0)]],
                          uint2 gid [[thread_position_in_grid]])
{
    // Check bounds
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) {
        return;
    }

    // Sobel 3x3 Kernels (Gx and Gy)
    float Gx[9] = {
        -1.0f, 0.0f, 1.0f,
        -2.0f, 0.0f, 2.0f,
        -1.0f, 0.0f, 1.0f
    };
    float Gy[9] = {
        -1.0f, -2.0f, -1.0f,
         0.0f,  0.0f,  0.0f,
         1.0f,  2.0f,  1.0f
    };

    int kernelDim = 3;
    int kernelRadius = kernelDim / 2;

    float2 uv = float2(gid) / float2(inTexture.get_width(), inTexture.get_height());
    float2 pixelSize = 1.0f / float2(inTexture.get_width(), inTexture.get_height());

    float4 gx_sum = float4(0.0f);
    float4 gy_sum = float4(0.0f);

    // Apply both Gx and Gy kernels
    for (int y = 0; y < kernelDim; ++y) {
        for (int x = 0; x < kernelDim; ++x) {
            int offsetX = x - kernelRadius;
            int offsetY = y - kernelRadius;
            float2 neighborUV = uv + float2(offsetX, offsetY) * pixelSize;

            // Read neighbor pixel (convert to grayscale for edge detection)
            float4 neighborColor = inTexture.sample(imageSampler, neighborUV);
            float luminance = 0.2126f * neighborColor.r + 0.7152f * neighborColor.g + 0.0722f * neighborColor.b;

            // Accumulate weighted luminance for Gx and Gy
            int kernelIndex = y * kernelDim + x;
            gx_sum.r += luminance * Gx[kernelIndex]; // Apply to one channel
            gy_sum.r += luminance * Gy[kernelIndex];
        }
    }

    // Calculate gradient magnitude: sqrt(Gx^2 + Gy^2)
    float magnitude = sqrt(gx_sum.r * gx_sum.r + gy_sum.r * gy_sum.r);

    // Read original alpha
    float4 centerColor = inTexture.read(gid);

    // Output edge magnitude as grayscale (clamped), preserve alpha
    float4 edgeColor = saturate(float4(magnitude, magnitude, magnitude, centerColor.a));

    outTexture.write(edgeColor, gid);
}

