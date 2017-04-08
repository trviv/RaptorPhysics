#ifndef __CLOTH
#define __CLOTH

#include "PhysicalEntity.h"

/// Class representing a cloth
class Cloth : public PhysicalEntity
{
public:
  void init(const Matrix4& transform, const float dim[],
    const Counter subdivision[]);
  void render();
  void step();
};

#endif