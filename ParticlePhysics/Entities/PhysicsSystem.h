#ifndef PHYSICS_SYSTEM_H
#define PHYSICS_SYSTEM_H

#include "../Solvers/Solver.h"
#include "PhysicsEntity.h"

/*
@class Class representing a system simulating physical entities.
*/
class PhysicsSystem : protected ShaderEntity, public Window
{
  /* Compute interface on which the system will operate on.*/
  ComputeInterface*               compute;

  /* Memory sections which needs updation.*/
  vector<SectionData>             updates;

  /* Entities in the system.*/
  vector<PhysicsEntity*>          entities;

  /* Shared data for entities.*/
  vector<ParticleSharedData*>     entitySharedData;

  /* Entities in the system.*/
  vector<vector<ParticleStruct>*> entityParticles;

  /* Memory allocators used by the system.*/
  vector<SharedAllocator*>        allocators;

  /* ushort solvers in the system.*/
  Solver<ushort, real, Real3>*    solversUshort[SOLVER_MAX];

  /* uint solvers in the system.*/
  Solver<uint, real, Real3>*      solversUint[SOLVER_MAX];

  /* Number of nodes in the system.*/
  uint  nodeCount;

  /* Total entities in the system.*/
  uint  totalEntityCount;

  /* Entity ids available for reuse.*/
  vector<uint>  availableEntityIds;

  void step();

  uint getNewEntityId();

public:

  PhysicsSystem(ComputeInterface* compute);

  ~PhysicsSystem();

  void init(int argc, char** argv, int width = 512, int height = 512,
    const char* name = "GL Window");

  void registerEntity(PhysicsEntity* entity);

#ifdef ENABLE_RENDERING

  const ParticleSharedData* getEntitySharedData(PhysicsEntity* entity)const;

  void render();
#endif

  void step(float timeStep);
};

#endif