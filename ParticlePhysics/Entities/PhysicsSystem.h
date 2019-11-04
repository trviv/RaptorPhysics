#ifndef PHYSICS_SYSTEM_H
#define PHYSICS_SYSTEM_H

#include "../Solvers/Solver.h"
#include "PhysicsEntity.h"
#include "../Solvers/Collision/CollisionSolver.h"

/*!
@class Class representing a system simulating physical entities.
*/
class PhysicsSystem : protected ShaderEntity, public Window
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
  EntitySolver<uint, real, Real3>*    solversUint[SOLVER_MAX];

  float elapsedSimTime;
  int   frameCount;

  void createSphere(float radius);

  void createUnitBox();

  void createUnitCircle();

  /*!@function Take one simulation step.*/
  void step();

  /*!@function Get solver instance for a solver type.*/
  void* getSolver(SolverType type);

  /*!@function Perform integration and differentiation step.*/
  void positionUpdate(float timeStep);

  /*!@function Perform differentiation step.*/
  void differentiate(float timeStep);

  /*!@function Perform integration step.*/
  void integrate(float timeStep);

public:

  /*!
  @constructor Create a new physics system using a compute interface.
  @param compute Compute interface through which the system will operate.
  @param maxParticles Maximum number of particles in the system.
  */
  PhysicsSystem(ComputeInterface* compute, const uint maxParticles = 1024 * 16);

  /*!@destructor Dellocate a physics system.*/
  ~PhysicsSystem();

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

#ifdef ENABLE_RENDERING

  Texture displayPositionBuffer;
  Texture displayColorBuffer;
  Texture displayCollisionBuffer;
  Texture displayBoxBuffer;
  Texture displayGridBuffer;

  Vertex  displayParticleVertex;
  Vertex  displaySolidVertex;
  Vertex  displayFlatVertex;
  Vertex  displayBoxVertex;
  Vertex  displayLineVertex;

  Shader  displayParticleShader;
  Shader  displaySolidShader;
  Shader  displayFlatShader;
  Shader  displayBoxShader;
  Shader  displayGridShader;
  Shader  displayLineShader;

  Face    displayParticleElements;
  Face    displayBoxElements;
  Face    displayGridElements;

  float   elapsedRenderTime;

  /*!@member Particle radius available for reuse.*/
  vector<float> solverParticleRadius[SOLVER_MAX];

  /*!@function Render all registered entities.*/
  void render();

#endif

  /*!@function Take one simulation step using the time step.*/
  void step(float timeStep);

  /*!@function Set acceleration due to gravity for the system.*/
  void setGravity(const Real3& gravity);

  /*!@function Set system bounding box of the system.*/
  void setSystemBoundary(const XAB& bound);
};

#endif
