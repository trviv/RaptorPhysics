#ifndef PHYSICS_ENTITY_H
#define PHYSICS_ENTITY_H

#include <Core.h>
#include "../Solvers/Solver.h"
#include "../Common/ParticleStruct.h"

enum PhysicsEntityType
{
  PHYSICS_ENTITY_CLOTH,
  PHYSICS_ENTITY_MAX
};

/*!
@class Base class for all physical entites
*/
class PhysicsEntity : protected SolverData<uint, real, Real3>
{
protected:

  Vertex  displayVertex;
  Shader  displayShader;
  Face    displayElements;

  uint    instanceCount;
  uint    solver;

  friend class PhysicsSystem;

public:

  PhysicsEntity();

  virtual void init(const Matrix4& transform,
    const real dim[],
    const uint subdivision[],
    const real mass) = 0;

  virtual void init(const Matrix4& transform,
    const real dim[],
    const real particleRadius,
    const real mass) = 0;

#ifdef ENABLE_RENDERING
  virtual void render(ParticleStruct* particles) = 0;
#endif
};

#endif