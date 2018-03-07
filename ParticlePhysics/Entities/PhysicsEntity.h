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

  uint    identity;

#ifdef ENABLE_RENDERING

  Vertex  displayVertex;
  Shader  displayShader;
  Face    displayElements;

#endif

  /*@member Entity solver type.*/
  SolverType  solver;

  /*@member Flag indicating wich section data is shared between instances.*/
  bool        sectionShared[SECTION_DATA_MAX];

  friend class PhysicsSystem;

public:

  PhysicsEntity();

  void setIdentity(uint instanceCount, uint entityId)
  {
    identity = (instanceCount << PHYSICS_INSTANCE_ID_SHIFT) | (entityId & PHYSICS_ENTITY_ID_MASK);
  }

#ifdef ENABLE_RENDERING
  virtual void render(ParticleStruct* particles) = 0;
#endif
};

#endif