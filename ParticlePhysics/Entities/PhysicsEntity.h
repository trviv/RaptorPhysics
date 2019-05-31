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

#ifdef ENABLE_RENDERING

  Vertex  displayVertex;
  Shader  displayShader;
  Face    displayElements;

#endif

  /*@member Entity solver type.*/
  SolverType  solver;

  friend class PhysicsSystem;

public:

  IdentityInfo identity;

  PhysicsEntity();

#ifdef ENABLE_RENDERING
  virtual void render(ParticleStruct* particles) = 0;
#endif
};

#endif
