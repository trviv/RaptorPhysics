#ifndef __RIGID_BODY
#define __RIGID_BODY

#include "PhysicalEntity.h"

/// Class representing a cloth
class RigidBody : public PhysicalEntity
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