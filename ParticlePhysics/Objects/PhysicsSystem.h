#ifndef __PHYSICS_SYSTEM
#define __PHYSICS_SYSTEM

#include <Core.h>
#include "../ConstraintSolver/DistanceConstrain.h"
#include "../ConstraintSolver/RigidConstrain.h"

class PhysicalEntity;

/// Class representing a cloth
CDEF class PhysicsSystem : public Window
{
  ConstrainSolver < __int32, real, Real3 > constrain;
  std::vector<PhysicalEntity*> objects;

public:
  ~PhysicsSystem();
  void init(int argc, char** argv, int width = 512, int height = 512,
    const char* name = "GL Window");
  void render();
  void step();

  DistanceConstrain* distanceConstrain()
  {
    return &(DistanceConstrain&)constrain;
  }

  const DistanceConstrain* distanceConstrain()const
  {
    return &(DistanceConstrain&)constrain;
  }

  RigidConstrain* rigidConstrain()
  {
    return &(RigidConstrain&)constrain;
  }

  const RigidConstrain* rigidConstrain()const
  {
    return &(RigidConstrain&)constrain;
  }
};

extern PhysicsSystem* physics_system;

#endif