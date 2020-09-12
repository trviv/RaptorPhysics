#ifndef PHYSICS_ENTITY_H
#define PHYSICS_ENTITY_H

#include <Core.h>
#include "../Solvers/Solver.h"
#include "../Common/ParticleStruct.h"

#define ENABLE_RENDERING

/*!
@class Base class for all physical entites
*/
class PhysicsEntity : protected SolverData<uint, real, Real3>
{
protected:

  /*!@member Entity solver type.*/
  SolverType  solver;

  friend class PhysicsSystem;

public:

#ifdef ENABLE_RENDERING
  Face displayElements;
  Face displayEdges;
#endif

  IdentityInfo identity;

  PhysicsEntity();

  virtual ~PhysicsEntity();
};

#endif
