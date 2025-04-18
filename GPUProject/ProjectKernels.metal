//
//  ProjectKernels.metal
//  GPUProject
//
//  Created by Arvin Ziaei on 2025-04-16.
//

#include <metal_stdlib>
using namespace metal;


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


