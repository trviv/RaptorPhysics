#ifndef __PHYSICS_SYSTEM
#define __PHYSICS_SYSTEM

#include <Core.h>
#include "../ConstraintSolver/DistanceConstrain.h"
#include "../ConstraintSolver/RigidConstrain.h"

class PhysicalEntity;

/// Class representing a cloth
CDEF class PhysicsSystem : public Window
{
  DistanceConstrain distance_constrain;
  RigidConstrain    rigid_constrain;
  std::vector<PhysicalEntity*> objects;

public:
  ~PhysicsSystem();

  void init(int argc, char** argv, int width = 512, int height = 512,
    const char* name = "GL Window");

  void render();

  void step();

  DistanceConstrain* distanceConstrain()
  {
    return &distance_constrain;
  }

  const DistanceConstrain* distanceConstrain()const
  {
    return &distance_constrain;
  }

  RigidConstrain* rigidConstrain()
  {
    return &rigid_constrain;
  }

  const RigidConstrain* rigidConstrain()const
  {
    return &rigid_constrain;
  }
};

extern PhysicsSystem* physics_system;

#endif