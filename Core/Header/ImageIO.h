#ifndef IMAGE_H
#define IMAGE_H

#include "ComputeInterface.h"
#include "GLClass.h"

class ImageIO
{
public:

  bool loadFile(const char *nameWithoutExtension, Texture *texture);

  bool loadFile(const char *nameWithoutExtension, Texture *texture, uint resizeWidth, uint resizeHeight);
};

#endif
