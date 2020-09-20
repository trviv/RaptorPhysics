#include <SDL2/SDL_atomic.h>
#include "ComputeGraphicsSharedTexture.h"

typedef struct
{
  int                 cvPixelFormat;
  SharedTextureFormat metalFormat;
  GLuint              glInternalFormat;
  GLuint              glFormat;
  GLuint              glType;
} MetalOpenGLTextureFormatInfo;

uint getBytesPerPixel(SharedTextureFormat pixelFormat)
{
  switch(pixelFormat)
  {
    case SHARED_TEXTURE_FORMAT_UINT8x4:
      return 4;
    case SHARED_TEXTURE_FORMAT_FLOAT16x4:
      return 8;
    case SHARED_TEXTURE_FORMAT_FLOAT32:
      return 4;
    case SHARED_TEXTURE_FORMAT_FLOAT32x2:
      return 8;
    case SHARED_TEXTURE_FORMAT_FLOAT32x4:
      return 16;
  }
  return 0;
}

TextureFormat getGraphicsTextureFormat(SharedTextureFormat pixelFormat)
{
  switch (pixelFormat) {
    case SHARED_TEXTURE_FORMAT_UINT8x4:
      return TEXTURE_FORMAT_UBYTE;
    case SHARED_TEXTURE_FORMAT_FLOAT16x4:
      return TEXTURE_FORMAT_FLOAT;
    case SHARED_TEXTURE_FORMAT_FLOAT32:
      return TEXTURE_FORMAT_FLOAT;
    case SHARED_TEXTURE_FORMAT_FLOAT32x2:
      return TEXTURE_FORMAT_FLOAT;
    case SHARED_TEXTURE_FORMAT_FLOAT32x4:
      return TEXTURE_FORMAT_FLOAT;
  }
  
  return TEXTURE_FORMAT_UBYTE;
}

#if defined(__APPLE__) && defined(USE_METAL_COMPUTE)

// Table of equivalent formats across CoreVideo, Metal and OpenGL
static const MetalOpenGLTextureFormatInfo AppleTextureFormatTable[] =
{
  // Core Video Pixel Format,               Metal Pixel Format,               GL internalformat,  GL format,      GL type
#if TARGET_OS_IPHONE
  {kCVPixelFormatType_OneComponent32Float,  SHARED_TEXTURE_FORMAT_FLOAT32,    GL_R32F,            GL_RED,         GL_FLOAT},
  {kCVPixelFormatType_TwoComponent32Float,  SHARED_TEXTURE_FORMAT_FLOAT32x2,  GL_RG32F,           GL_RG,          GL_FLOAT},
  {kCVPixelFormatType_32BGRA,               SHARED_TEXTURE_FORMAT_UINT8x4,    GL_RGBA,            GL_BGRA,        GL_UNSIGNED_BYTE},
  {kCVPixelFormatType_64RGBAHalf,           SHARED_TEXTURE_FORMAT_FLOAT16x4,  GL_RGBA16F,         GL_RGBA,        GL_HALF_FLOAT},
  {kCVPixelFormatType_128RGBAFloat,         SHARED_TEXTURE_FORMAT_FLOAT32x4,  GL_RGBA32F,         GL_RGBA,        GL_FLOAT},
#else
  {kCVPixelFormatType_OneComponent32Float,  SHARED_TEXTURE_FORMAT_FLOAT32,    GL_RED,             GL_RED,         GL_FLOAT},
  {kCVPixelFormatType_TwoComponent32Float,  SHARED_TEXTURE_FORMAT_FLOAT32x2,  GL_RG32F,           GL_RG,          GL_FLOAT},
  {kCVPixelFormatType_32BGRA,               SHARED_TEXTURE_FORMAT_UINT8x4,    GL_RGBA,            GL_BGRA,        GL_UNSIGNED_BYTE},
  {kCVPixelFormatType_64RGBAHalf,           SHARED_TEXTURE_FORMAT_FLOAT16x4,  GL_RGBA16F,         GL_RGBA,        GL_HALF_FLOAT},
  {kCVPixelFormatType_128RGBAFloat,         SHARED_TEXTURE_FORMAT_FLOAT32x4,  GL_RGBA32F,         GL_RGBA,        GL_FLOAT},
#endif
};

const MetalOpenGLTextureFormatInfo* textureFormatInfoFromMetalPixelFormat(SharedTextureFormat pixelFormat)
{
  for (int i=0; i<sizeof(AppleTextureFormatTable)/sizeof(MetalOpenGLTextureFormatInfo); i++)
  {
    if (pixelFormat == AppleTextureFormatTable[i].metalFormat)
    {
      return &AppleTextureFormatTable[i];
    }
  }
  return NULL;
}

ComputeGraphicsSharedTexture::ComputeGraphicsSharedTexture():compute(NULL)
{}

ComputeGraphicsSharedTexture::ComputeGraphicsSharedTexture(ComputeInterface* compute, GLContext* glContext, SharedTextureFormat textureFormat, uint size[2]):graphicsRef(getGraphicsTextureFormat(textureFormat))
{
  this->compute = compute;
  CVPixelBufferRef          cvPixelBuffer = NULL;
  CVMetalTextureRef         cvMetalTexture;
  CVMetalTextureCacheRef    cvMetalTextureCache = NULL;
#if TARGET_OS_OSX
  CVOpenGLTextureRef        cvGlTexture;
  CVOpenGLTextureCacheRef   cvGlTextureCache;
#else
  CVOpenGLESTextureRef      cvGlTexture;
  CVOpenGLESTextureCacheRef cvGlTextureCache;
#endif

#if TARGET_OS_OSX
  graphicsRef.setBindType(GL_TEXTURE_RECTANGLE);
#endif

  graphicsRef.init(size[0], size[1]);

  const MetalOpenGLTextureFormatInfo* format = textureFormatInfoFromMetalPixelFormat(textureFormat);

  if (!format)
  {
    logComputeError("Metal format supplied not supported in this sample");
  }

  NSDictionary* cvBufferProperties = @{
    (NSString*)kCVPixelBufferOpenGLCompatibilityKey: @YES,
    (NSString*)kCVPixelBufferMetalCompatibilityKey: @YES
  };

  CVReturn cvRet = CVPixelBufferCreate(kCFAllocatorDefault, size[0], size[1], format->cvPixelFormat, (__bridge CFDictionaryRef)cvBufferProperties, &cvPixelBuffer);

  if (cvRet != kCVReturnSuccess)
  {
    logComputeError("Failed to create CVPixelBuffer");
  }

#if TARGET_OS_OSX
  // 1. Create an OpenGL CoreVideo texture cache from the pixel buffer.
  cvRet  = CVOpenGLTextureCacheCreate(kCFAllocatorDefault, NULL, glContext.CGLContextObj, glContext.pixelFormat.CGLPixelFormatObj, NULL, &cvGlTextureCache);
  if (cvRet != kCVReturnSuccess)
  {
    logComputeError("Failed to create OpenGL texture cache");
  }

  // 2. Create a CVPixelBuffer-backed OpenGL texture image from the texture cache.
  cvRet = CVOpenGLTextureCacheCreateTextureFromImage(kCFAllocatorDefault, cvGlTextureCache, cvPixelBuffer, NULL, &cvGlTexture);
  if (cvRet != kCVReturnSuccess)
  {
    logComputeError("Failed to create OpenGL texture from image");
  }

  // 3. Get an OpenGL texture name from the CVPixelBuffer-backed OpenGL texture image.
  graphicsRef.set(CVOpenGLTextureGetName(cvGlTexture));

#else
  // 1. Create an OpenGL ES CoreVideo texture cache from the pixel buffer.
  cvRet = CVOpenGLESTextureCacheCreate(kCFAllocatorDefault, NULL, glContext, NULL, &cvGlTextureCache);
  if (cvRet != kCVReturnSuccess)
  {
    logComputeError("Failed to create OpenGL ES texture cache");
  }

  // 2. Create a CVPixelBuffer-backed OpenGL ES texture image from the texture cache.
  cvRet = CVOpenGLESTextureCacheCreateTextureFromImage(kCFAllocatorDefault, cvGlTextureCache,
                                                       cvPixelBuffer, NULL, GL_TEXTURE_2D,
                                                       format->glInternalFormat,
                                                       size[0], size[1],
                                                       format->glFormat,
                                                       format->glType, 0, &cvGlTexture);

  if (cvRet != kCVReturnSuccess)
  {
    logComputeError("Failed to create OpenGL ES texture from image");
  }

  // 3. Get an OpenGL ES texture name from the CVPixelBuffer-backed OpenGL ES texture image.
  graphicsRef.set(CVOpenGLESTextureGetName(cvGlTexture));

#endif

  // 1. Create a Metal Core Video texture cache from the pixel buffer.
  cvRet = CVMetalTextureCacheCreate(kCFAllocatorDefault, NULL, compute->getDevice(), NULL, &cvMetalTextureCache);
  if (cvRet != kCVReturnSuccess)
  {
    logComputeError("Failed to create Metal texture cache");
  }

  // 2. Create a CoreVideo pixel buffer backed Metal texture image from the texture cache.
  cvRet = CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, cvMetalTextureCache, cvPixelBuffer, NULL,
                                                    (MTLPixelFormat)format->metalFormat, size[0], size[1], 0, &cvMetalTexture);

  if (cvRet != kCVReturnSuccess)
  {
    logComputeError("Failed to create Metal texture from image");
  }

  ComputeTextureIdentifier textureIdentifier = CVMetalTextureGetTexture(cvMetalTexture);

  // 3. Get a Metal texture using the CoreVideo Metal texture reference.
  // Get a Metal texture object from the Core Video pixel buffer backed Metal texture image
  if (!textureIdentifier)
  {
    logComputeError("Failed to get metal texture from CVMetalTextureRef");
  }

  computeRef = ComputeTexture(textureIdentifier, (uint[2]){size[0], size[1]}, getBytesPerPixel(textureFormat));
}

ComputeTexture& ComputeGraphicsSharedTexture::getComputeTexture()
{
  return computeRef;
}

Texture& ComputeGraphicsSharedTexture::getGraphicsTexture()
{
  return graphicsRef;
}

#else

ComputeGraphicsSharedTexture::ComputeGraphicsSharedTexture():compute(NULL)
{}

ComputeGraphicsSharedTexture::ComputeGraphicsSharedTexture(ComputeInterface* compute, GLContext* glContext, SharedTextureFormat textureFormat, uint size[2])
{}

ComputeTexture& ComputeGraphicsSharedTexture::getComputeTexture()
{
  logComputeError("ComputeGraphicsSharedTexture is not supported with OpenCL");
  return computeRef;
}

Texture& ComputeGraphicsSharedTexture::getGraphicsTexture()
{
  logComputeError("ComputeGraphicsSharedTexture is not supported with OpenCL");
  return graphicsRef;
}

#endif
