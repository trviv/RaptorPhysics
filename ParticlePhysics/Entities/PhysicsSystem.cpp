#include "PhysicsSystem.h"

#include "../Solvers/LinearSolver.h"
#include "../Solvers/DistanceSolver.h"
#include "../Solvers/RigidSolver.h"

PhysicsSystem::PhysicsSystem(ComputeInterface* compute)
  : compute(compute)
{
  nodeCount = 0;
  totalEntityCount = 0;
  availableEntityIds.clear();

  for (uint i = 0; i < SOLVER_MAX; i++)
  {
    solversUshort[i] = NULL;
    solversUint[i] = NULL;
  }

  includeFiles.push_back("ComputeHeader.shader");
  includeFiles.push_back("ParticleStruct.h");

  vector<string> newType = { "uint", "float", "float3" };
  vector<string> oldType = { "IndexType", "CoefficientType", "VariableType" };
  registerShader(compute, "PhysicsSystem.shader", &oldType, &newType);
  kernels.push_back(programs[0].createKernel("integrate"));

  solverEntityOffsets.create(compute, NULL, true);
  solverNodeOffsets.create(compute, NULL, true);

  solverEntityOffsets.resize(SOLVER_MAX, false);
  solverNodeOffsets.resize(SOLVER_MAX, false);

  solverEntityOffsets.host()->resize(SOLVER_MAX);
  solverNodeOffsets.host()->resize(SOLVER_MAX);

  for (uint i = 0; i < SOLVER_MAX; i++)
  {
    (*solverEntityOffsets.host())[i] = 0;
    (*solverNodeOffsets.host())[i] = 0;
  }
}

PhysicsSystem::~PhysicsSystem()
{
  for (uint i = 0; i < entities.size(); i++)
  {
    delete entities[i];
    entities[i] = NULL;
  }
}

/*bool compareEntity(PhysicsEntity* i, PhysicsEntity* j)
{
return (i->getId() < j->getId());
}*/

void* PhysicsSystem::getSolver(SolverType type)
{
  // get solver
  int index = mCeilExpOf2((uint)type);

  if (!solversUint[index])
  {
    switch (type)
    {
    case SOLVER_NULL:
    case SOLVER_EQUATION:
      break;
    case SOLVER_CLOTH:
    {
      solversUint[index] = new DistanceSolver(compute, allocators[0]);
      break;
    }
    case SOLVER_RIGID_BODY:
    {
      solversUint[index] = new RigidSolver(compute, allocators[0]);
      break;
    }
    default:
      printf("Undefined!");
      assert(0);
      break;
    }
  }

  return solversUint[index];
}

void PhysicsSystem::registerEntity(PhysicsEntity* entity)
{
  Matrix4 iden;
  iden.setIdentity();
  registerEntity(entity, 1, &iden);
}

void PhysicsSystem::registerEntity(PhysicsEntity* entity, const ushort instanceCount, const Matrix4* instanceTransforms)
{
  // create memory heap allocators for the system
  if (!allocators.size())
  {
    SharedAllocator* allocator = new SharedAllocator(compute);
    int multiplier = 2048;
    allocator->particleAllocator.create(multiplier * 1024);
    allocator->constrainAllocator.create(multiplier * 1024, 4096);
    allocators.push_back(allocator);
  }

  Solver<uint, real, Real3>* solver = (Solver<uint, real, Real3>*)getSolver(entity->solver);

  // create entity section data
  SectionData sectionData;
  SectionData updateInfo;

  sectionData.offsets[SECTION_DATA_NODE] = solver->nodeOffset;
  sectionData.offsets[SECTION_DATA_CONNECTION] = solver->connectionOffset;

  // add entity shared data to the system
  solver->particleSharedData.host()->push_back(entity->particleSharedData.host()->at(0));

  // update information
  updateInfo.offsets[SECTION_DATA_NODE] = nodeCount;

  uint entityId = solver->newEntityId();
  entity->identity.setIdentity(instanceCount, entityId);
  entity->identity.setSolver(entity->solver);
  sectionData.identity = entity->identity;

  for (uint i = 0; i < instanceCount; i++)
  {
    if ((i == 0) || (!entity->sectionShared[SECTION_DATA_CONNECTION]))
    {
      solver->rawConstrainConnections.insert(solver->rawConstrainConnections.end(),
        entity->rawConstrainConnections.begin(), entity->rawConstrainConnections.end());

      solver->rawConstrainCoefficients.insert(solver->rawConstrainCoefficients.end(),
        entity->rawConstrainCoefficients.begin(), entity->rawConstrainCoefficients.end());
    }

    if ((i == 0) || (!entity->sectionShared[SECTION_DATA_NODE]))
    {
      solver->constrainConstants.host()->insert(solver->constrainConstants.host()->end(),
        entity->constrainConstants.host()->begin(), entity->constrainConstants.host()->end());

      solver->particleAuxData.host()->insert(solver->particleAuxData.host()->end(),
        entity->particleAuxData.host()->begin(), entity->particleAuxData.host()->end());

      solver->particleRigidData.host()->insert(solver->particleRigidData.host()->end(),
        entity->particleRigidData.host()->begin(), entity->particleRigidData.host()->end());
    }

    vector<Real3>* entityPositions = entity->constrainConstants.host();
    for (uint p = 0; p < entityPositions->size(); p++)
    {
      ParticleStruct particle;
      instanceTransforms[i].transformPos(particle.position, entityPositions->at(p));

      IdentityInfo particleIdentity;
      particleIdentity.setIdentity(i, entityId);
      particleIdentity.setSolver(entity->solver);

      solver->particles.host()->push_back(particle);
      solver->particleIdentities.host()->push_back(particleIdentity);
    }

    solver->commit();
  }

  nodeCount += solver->nodeOffset;

  updateInfo.counts[SECTION_DATA_NODE] = nodeCount - updateInfo.offsets[SECTION_DATA_NODE];

  sectionData.counts[SECTION_DATA_NODE] = solver->nodeOffset - sectionData.offsets[SECTION_DATA_NODE];
  sectionData.counts[SECTION_DATA_CONNECTION] = solver->connectionOffset - sectionData.offsets[SECTION_DATA_CONNECTION];

  solver->deviceSections.host()->push_back(sectionData);

  uint cumulativeOffset = 0;
  uint cumulativeEntities = 0;
  // update starting offset for each solver
  for (uint i = 1; i < SOLVER_MAX; i++)
  {
    Solver<uint, real, Real3>* localSolver = (Solver<uint, real, Real3>*)getSolver((SolverType)(1 << (i - 1)));
    if (localSolver)
    {
      cumulativeOffset += localSolver->nodeOffset;
      cumulativeEntities += localSolver->entityCount();
    }
    (*solverEntityOffsets.host())[i] = cumulativeEntities;
    (*solverNodeOffsets.host())[i] = cumulativeOffset;
  }
  solverEntityOffsets.syncDevice();
  solverNodeOffsets.syncDevice();

  // register entity properties
  entities.push_back(entity);
  entityParticles.push_back(solver->particles.host());
  entitySectionData.push_back(sectionData);
  updates.push_back(updateInfo);
}

void PhysicsSystem::step()
{
  ProfileManager::Reset();

  if (updates.size())
  {
    ProfileBlock("Physics system update");
    SharedAllocator* allocator = allocators[0];

    for (const SectionData& section : updates)
    {
      // reset position delta for entity
      float zero = 0;

      compute->setBuffer(allocator->getHeap(COMPUTE_HEAP_PARTICLE_DELTA)->get(),
        section.offsets[SECTION_DATA_NODE] * sizeof(ParticleStruct),
        section.counts[SECTION_DATA_NODE] * sizeof(ParticleStruct),
        &zero, sizeof(float));

      compute->setBuffer(allocator->getHeap(COMPUTE_HEAP_PARTICLE_DIFF)->get(),
        section.offsets[SECTION_DATA_NODE] * sizeof(ParticleDifferential),
        section.counts[SECTION_DATA_NODE] * sizeof(ParticleDifferential),
        &zero, sizeof(float));
    }
    updates.clear();
  }
  step(.066f);

  compute->sync();
  ProfileManager::dumpAll(stdout);
  ProfileManager::Increment_Frame_Counter();
}

#ifdef ENABLE_RENDERING

/*const ParticleSharedData* PhysicsSystem::getEntitySharedData(PhysicsEntity* entity)const
{
const vector<PhysicsEntity*>::const_iterator entityLocation = find(entities.begin(), entities.end(), entity);
if (entityLocation != entities.end())
{
printf("Entity not found!");
assert(0);
}
return entitySharedData[(entityLocation - entities.begin())];
}*/

void PhysicsSystem::render()
{
  for (uint i = 0; i < SOLVER_MAX; i++)
  {
    if (solversUint[i])
    {
      uint elements = solversUint[i]->nodes();
      if (!elements) continue;
      solversUint[i]->particles.syncHost(0, elements);
    }
  }

  for (uint i = 0; i < entitySectionData.size(); i++)
  {
    uint entityOffset = entitySectionData[i].offsets[SECTION_DATA_NODE];
    uint solverId = getSolverId(entitySectionData[i].identity);
    if (solverId)
    {
      entityOffset += solverNodeOffsets.host()->at(solverId - 1);
    }
    entities[i]->render(&(*entityParticles[i])[entityOffset]);
  }
}

#endif

void PhysicsSystem::step(float timeStep)
{
  ProfileBlock("Physics system step");

  for (uint i = 0; i < SOLVER_MAX; i++)
  {
    if (solversUint[i])
    {
      solversUint[i]->solve();
    }
  }

  // block to integrate
  {
    SharedAllocator* allocator = allocators[0];

    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, nodeCount);

    ComputeMemory* buffers[] = {
      allocator->getHeap(COMPUTE_HEAP_PARTICLE)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_IDENTITY)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_DELTA)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_DIFF)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_SHARED)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_AUX)->get(),
      solverEntityOffsets.device(),
      solverNodeOffsets.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[0].setArgs(buffers, bufferCount);
    kernels[0].setArg<float>(&timeStep, bufferCount);
    kernels[0].setArg<uint>(&nodeCount, bufferCount + 1);
    compute->execute(kernels[0], workgroupSize, workgroupCount);
  }
}