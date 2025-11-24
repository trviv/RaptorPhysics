/*
 * Copyright (c) Vivek Trivedi, 2025
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#include "ImageLoader.h"
#include "GLClass.h"
#include <Utils/IOInterface.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb/stb_image_resize.h>

bool ImageLoader::checkImageExist(const char* fileName)
{
  return IOInterface::checkFileExist((fileName + string(".png")).c_str());
}

bool ImageLoader::readImageFile(const char *nameWithoutExtension, Texture *texture)
{
  int imageWidth = 0;
  int imageHeight = 0;
  int comp;

  string fileData = IOInterface::readFile((nameWithoutExtension+string(".png")).c_str());
  unsigned char* imageData = stbi_load_from_memory((unsigned char*)fileData.c_str(), (int)fileData.size(), &imageWidth, &imageHeight, &comp, 4);

  if (imageData == NULL)
  {
    return false;
  }

  *texture = Texture(TEXTURE_FORMAT_UBYTE);

  texture->init(imageWidth, imageHeight);
  texture->gen((float*)imageData, Texture::COLOR_BUFFER_UCHAR);

  stbi_image_free(imageData);

  return true;
}

bool ImageLoader::readImageFile(const char *nameWithoutExtension, Texture *texture, uint resizeWidth, uint resizeHeight)
{
  int imageWidth = 0;
  int imageHeight = 0;
  int comp;

  string fileData = IOInterface::readFile((nameWithoutExtension+string(".png")).c_str());
  unsigned char* imageData = stbi_load_from_memory((unsigned char*)fileData.c_str(), (int)fileData.size(), &imageWidth, &imageHeight, &comp, 4);

  if (imageData == NULL)
  {
    return false;
  }

  *texture = Texture(TEXTURE_FORMAT_UBYTE);

  vector<unsigned char> imageData2;
  imageData2.resize(resizeWidth*resizeHeight*4);
  stbir__resize_arbitrary(NULL, imageData, imageWidth, imageHeight, 0, &imageData2[0], resizeWidth, resizeHeight, 0,
                          0, 0, 1, 1, NULL,
                          4, -1, 0, STBIR_TYPE_UINT8, STBIR_FILTER_DEFAULT, STBIR_FILTER_DEFAULT,
                          STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP, STBIR_COLORSPACE_LINEAR);
  stbi_image_free(imageData);

  texture->init(resizeWidth, resizeHeight);
  texture->gen((float*)&imageData2[0], Texture::COLOR_BUFFER_UCHAR);

  return true;
}
