#ifndef PHYSICS_SYSTEM_H
#define PHYSICS_SYSTEM_H

#include "../Solvers/Solver.h"
#include "PhysicsEntity.h"
#include "../Solvers/Collision/CollisionSolver.h"

/*!
@class Class representing a system simulating physical entities.
*/
class PhysicsSystem : protected ShaderEntity
{
  /*!@member Compute interface through which the system will operate.*/
  ComputeInterface*               compute;

  /*!@member Memory sections which needs updation.*/
  vector<EntityLocation>          updates;

  /*!@member Memory allocators used by the system.*/
  vector<SharedAllocator*>        allocators;

  /*!@member Collision solver for the system.*/
  CollisionSolver*                collisionSolver;

  /*!@member Registered entities.*/
  vector<PhysicsEntity*>          entities[SOLVER_MAX];

  /*!@member Number of unique nodes in the system.*/
  uint                            nodeCount;

  /*!@member Number of instanced nodes in the system.*/
  uint                            instanceNodeCount;

  /*!@member Entity ids available for reuse.*/
  vector<uint>                    availableEntityIds;

  /*!@member Thread index map to absolute node index.*/
  DeviceArray<uint>               indexMap;

  /*!@member Physics system settings.*/
  DeviceArray<PhySystemSettings>  systemSettings;

  /*!@member ushort solvers in the system.*/
  EntitySolver<ushort, real, Real3>*  solversUshort[SOLVER_MAX];

  /*!@member uint solvers in the system.*/
  EntitySolverType*                   solversUint[SOLVER_MAX];

  /*!@function Take one simulation step using the time step.*/
  void step(float timeStep);

  /*!@function Get and Init if necessary solver instance for a solver type.*/
  EntitySolverType* getAndInitSolver(SolverType type);

  /*!@function Perform integration and differentiation step.*/
  void positionUpdate(float timeStep);

  /*!@function Perform differentiation step.*/
  void differentiate(float timeStep);

  /*!@function Perform integration step.*/
  void integrate(float timeStep);

  int simulationIterations;
  int solverIterations;

  friend class ReaderScene;
public:

  /*!
  @function Initialize physics system using a compute interface and max particles.
  @param compute Compute interface through which the system will operate.
  @param maxParticles Maximum number of particles in the system.
  */
  void init(ComputeInterface* compute, const uint maxParticles);

  /*!@destructor Dellocate a physics system.*/
  ~PhysicsSystem();

  /*!@function Get total number of particles in the system.*/
  uint particleCount()const;

  /*!@function Get collision solver for the system.*/
  const CollisionSolver* getCollisionSolver()const;

  /*!@function Get system physics system settings.*/
  const PhySystemSettings& getSystemSettings()const;

  /*!@function Get entities in a solver.*/
  vector<PhysicsEntity*>& getEntities(SolverType type);

  /*!@function Get solver instance for a solver type.*/
  EntitySolverType* getSolver(SolverType type);

  /*!
  @function Register a physics entity to the system.
  @param entity Entity to register.
  @return Identifier for the registered entity.
  */
  PhysicsEntityId registerEntity(PhysicsEntity* entity);

  /*!
  @function Add instance(s) of the physics entity to the system.
  @param registeredEntityId Entity to register.
  @param instanceCount Number of entity instances.
  @param instanceTransform Initial world transform for instances.
  */
  void addEntityInstance(const PhysicsEntityId registeredEntityId, const ushort instanceCount, const Matrix4* instanceTransforms);

  /*!@function Take one simulation step.*/
  void step();

  /*!@function Set acceleration due to gravity for the system.*/
  void setGravity(const Real3& gravity);

  /*!@function Set system bounding box of the system.*/
  void setSystemBoundary(const XAB& bound);
};

#endif
