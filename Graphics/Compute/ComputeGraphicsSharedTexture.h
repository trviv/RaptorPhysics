/*
 * Copyright (c) Vivek Trivedi, 2025
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#ifndef COMPUTE_GRAPHICS_SHARED_TEXTURE
#define COMPUTE_GRAPHICS_SHARED_TEXTURE

#include "../Graphics/GLClass.h"
#include <Compute/ComputeInterface.h>

#ifdef __APPLE__

#if TARGET_OS_OSX
#import <AppKit/AppKit.h>
#define GLContext NSOpenGLContext
#else
#import <UIKit/UIKit.h>
#define GLContext EAGLContext
#endif

enum SharedTextureFormat
{
  SHARED_TEXTURE_FORMAT_UINT8x4   = MTLPixelFormatRGBA8Uint,
  SHARED_TEXTURE_FORMAT_FLOAT32   = MTLPixelFormatR32Float,
  SHARED_TEXTURE_FORMAT_FLOAT32x4 = MTLPixelFormatRGBA32Float,
  SHARED_TEXTURE_FORMAT_FLOAT32x2 = MTLPixelFormatRG32Float,
  SHARED_TEXTURE_FORMAT_FLOAT16x4 = MTLPixelFormatRGBA16Float,
};

#else

#define GLContext void

enum SharedTextureFormat
{
  SHARED_TEXTURE_FORMAT_UINT8x4,
  SHARED_TEXTURE_FORMAT_FLOAT32,
  SHARED_TEXTURE_FORMAT_FLOAT32x4,
  SHARED_TEXTURE_FORMAT_FLOAT32x2,
  SHARED_TEXTURE_FORMAT_FLOAT16x4,
};

#endif

/*!
@class Interface representing a texture shared between Graphics and Compute APIs.
*/
class ComputeGraphicsSharedTexture
{
  ComputeInterface *compute;
  ComputeTexture  computeRef;
  Texture         graphicsRef;
public:
  ComputeGraphicsSharedTexture();

  ComputeGraphicsSharedTexture(ComputeInterface* compute, GLContext* glContext, SharedTextureFormat textureFormat, uint size[2]);

  ComputeTexture& getComputeTexture();

  Texture& getGraphicsTexture();
};

#endif
