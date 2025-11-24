/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include <Header/Math.h>

class Texture;

class ImageLoader
{
public:

  static bool checkImageExist(const char* fileName);

  static bool readImageFile(const char *nameWithoutExtension, Texture *texture);

  static bool readImageFile(const char *nameWithoutExtension, Texture *texture, uint resizeWidth, uint resizeHeight);

};

#endif
