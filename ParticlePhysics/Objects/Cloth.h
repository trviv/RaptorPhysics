#ifndef __CLOTH
#define __CLOTH

#include "PhysicalEntity.h"

/// Class representing a cloth
class Cloth : public PhysicalEntity
{
public:
  void init(const Matrix4& transform, const real dim[],
    const Counter subdivision[], const real mass);
  void init(const Matrix4& transform, const real dim[],
    const real particle_radius, const real mass);
  void render();
  void step();
};

#endif