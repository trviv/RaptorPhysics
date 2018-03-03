#ifndef PHYSICS_SYSTEM_H
#define PHYSICS_SYSTEM_H

#include "../Solvers/Solver.h"
#include "PhysicsEntity.h"

/*
@class Class representing a system simulating physical entities.
*/
class PhysicsSystem : protected ShaderEntity, public Window
{
  /*@member Compute interface on which the system will operate on.*/
  ComputeInterface*               compute;

  /*@member Memory sections which needs updation.*/
  vector<SectionData>             updates;

  /*@member Entities in the system.*/
  vector<PhysicsEntity*>          entities;

  /*@member Entity data offsets.*/
  vector<SectionData>             entitySectionData;

  /*@member Entities in the system.*/
  vector<vector<ParticleStruct>*> entityParticles;

  /*@member Memory allocators used by the system.*/
  vector<SharedAllocator*>        allocators;

  /*@member ushort solvers in the system.*/
  Solver<ushort, real, Real3>*    solversUshort[SOLVER_MAX];

  /*@member uint solvers in the system.*/
  Solver<uint, real, Real3>*      solversUint[SOLVER_MAX];

  /*@member Number of nodes in the system.*/
  uint  nodeCount;

  /*@member Total entities in the system.*/
  uint  totalEntityCount;

  /*@member Entity ids available for reuse.*/
  vector<uint>  availableEntityIds;

  /*@function Take one simulation step.*/
  void step();

  /*@function Get an available unique entity id.*/
  uint getNewEntityId();

public:

  /*@constructor Create a new physics system using a compute interface.*/
  PhysicsSystem(ComputeInterface* compute);

  /*@destructor Dellocate a physics system.*/
  ~PhysicsSystem();

  /*@function Initialize a physics system.*/
  void init(int argc, char** argv, int width = 512, int height = 512,
    const char* name = "GL Window");

  /*
  @function Register a physics entity to the system.
  @param entity Entity to register.
  */
  void registerEntity(PhysicsEntity* entity);

#ifdef ENABLE_RENDERING

  /*
  @function Get shared data for a given entity.
  @param entity Entity to search.
  */
  //const ParticleSharedData* getEntitySharedData(PhysicsEntity* entity)const;

  /*@function Render all registered entities.*/
  void render();

#endif

  /*@function Take one simulation step using the time step.*/
  void step(float timeStep);
};

#endif