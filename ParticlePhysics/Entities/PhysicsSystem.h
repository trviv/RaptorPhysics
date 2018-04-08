#ifndef PHYSICS_SYSTEM_H
#define PHYSICS_SYSTEM_H

#include "../Solvers/Solver.h"
#include "PhysicsEntity.h"
#include "../Solvers/CollisionSolver.h"

enum GlobalOffsetsEnum
{
  GLOBAL_NODE_OFFSET,
  GLOBAL_INSTANCE_OFFSET,
  GLOBAL_SOLVER_OFFSET
};

struct uint4
{
  uint value[4];

  uint& operator[](int index)
  {
    return value[index];
  }

  const uint& operator[](int index)const
  {
    return value[index];
  }
};

/*
@class Class representing a system simulating physical entities.
*/
class PhysicsSystem : protected ShaderEntity, public Window
{
  /*@member Compute interface on which the system will operate on.*/
  ComputeInterface*               compute;

  /*@member Memory sections which needs updation.*/
  vector<EntityLocation>          updates;

  /*@member Memory allocators used by the system.*/
  vector<SharedAllocator*>        allocators;

  /*@member ushort solvers in the system.*/
  Solver<ushort, real, Real3>*    solversUshort[SOLVER_MAX];

  /*@member uint solvers in the system.*/
  Solver<uint, real, Real3>*      solversUint[SOLVER_MAX];

  /*@member Collision solver for the system.*/
  CollisionSolver                 collisionSolver;

  /*@member Registered entities.*/
  vector<PhysicsEntity*>          entities[SOLVER_MAX];

  /*@member Number of unique nodes in the system.*/
  uint                            nodeCount;

  /*@member Number of instanced nodes in the system.*/
  uint                            instanceNodeCount;

  /*@member Entity ids available for reuse.*/
  vector<uint>                    availableEntityIds;

  /*@member Offsets for different solvers.*/
  DeviceArray<uint4>              globalOffsets;

  /*@function Take one simulation step.*/
  void step();

  /*@function Get solver instance for a solver type.*/
  void* getSolver(SolverType type);

public:

  /*@constructor Create a new physics system using a compute interface.*/
  PhysicsSystem(ComputeInterface* compute);

  /*@destructor Dellocate a physics system.*/
  ~PhysicsSystem();

  /*
  @function Register a physics entity to the system.
  @param entity Entity to register.
  @return Identifier for the registered entity.
  */
  PhysicsEntityId registerEntity(PhysicsEntity* entity);

  /*
  @function Add instance(s) of the physics entity to the system.
  @param registeredEntityId Entity to register.
  @param instanceCount Number of entity instances.
  @param instanceTransform Initial world transform for instances.
  */
  void addEntityInstance(const PhysicsEntityId registeredEntityId, const ushort instanceCount, const Matrix4* instanceTransforms);

#ifdef ENABLE_RENDERING

  Texture displayPositionBuffer;
  Vertex  displayVertex;
  Shader  displayShader;
  Face    displayElements;

  void createSphere(float radius);

  /*@function Render all registered entities.*/
  void render();

#endif

  /*@function Take one simulation step using the time step.*/
  void step(float timeStep);
};

#endif