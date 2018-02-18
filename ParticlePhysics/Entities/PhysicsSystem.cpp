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

}

PhysicsSystem::~PhysicsSystem()
{
  for (uint i = 0; i < entities.size(); i++)
  {
    delete entities[i];
    entities[i] = NULL;
  }
}

void PhysicsSystem::init(int argc, char** argv, int width,
  int height, const char* name)
{
  Window::init(argc, argv, width, height);
}

/*bool compareEntity(PhysicsEntity* i, PhysicsEntity* j)
{
return (i->getId() < j->getId());
}*/

uint PhysicsSystem::getNewEntityId()
{
  return totalEntityCount++;
}

void PhysicsSystem::registerEntity(PhysicsEntity* entity)
{
  entities.push_back(entity);

  SharedAllocator* allocator = NULL;
  if (!allocators.size())
  {
    allocator = new SharedAllocator(compute);
    allocator->particleAllocator.create(1024);
    allocator->constrainAllocator.create(1024, 4096);
    allocators.push_back(allocator);
  }

  // get solver
  Solver<uint, real, Real3>* solver = NULL;
  if (!solversUint[entity->solver])
  {
    int index = mExpOf2(entity->solver);
    switch (entity->solver)
    {
    case SOLVER_CLOTH:
    {
      solversUint[index] = new DistanceSolver(compute, allocator);
      solver = solversUint[index];
      break;
    }
    case SOLVER_RIGID_BODY:
    {
      solversUint[index] = new RigidSolver(compute, allocator);
      solver = solversUint[index];
      break;
    }

    default:
      printf("Undefined!");
      assert(0);
      break;
    }
  }
  else
  {
    solver = solversUint[entity->solver];
  }

  // copy entity data to the solver

  // add entity shared data to the system
  solver->particleSharedData.host()->push_back(entity->particleSharedData.host()->at(0));

  // create entity header
  {
    ParticleSharedData* sharedData = &solver->particleSharedData.host()->back();
    sharedData->entityAlloc.offsets[DEVICE_HEADER_NODE] = solver->nodeOffset;
    sharedData->entityAlloc.offsets[DEVICE_HEADER_CONNECTION] = solver->connectionOffset;
    //sharedData->entityAlloc.counts[DEVICE_HEADER_NODE] = entity->nodeCount;
    //sharedData->entityAlloc.counts[DEVICE_HEADER_CONNECTION] = entity->connectionCount;
    entitySharedData.push_back(sharedData);
  }

  entityParticles.push_back(solver->particles.host());

  SectionData updateInfo;
  updateInfo.offsets[DEVICE_HEADER_NODE] = nodeCount;

  for (uint i = 0; i < entity->instanceCount; i++)
  {
    if (solver->rawConstrainConnections.size())
    {
      solver->rawConstrainConnections.insert(entity->rawConstrainConnections.begin(),
        entity->rawConstrainConnections.end(),
        solver->rawConstrainConnections.end());
    }
    else
    {
      solver->rawConstrainConnections = entity->rawConstrainConnections;
    }

    if (solver->rawConstrainCoefficients.size())
    {
      solver->rawConstrainCoefficients.insert(entity->rawConstrainCoefficients.begin(),
        entity->rawConstrainCoefficients.end(),
        solver->rawConstrainCoefficients.end());
    }
    else
    {
      solver->rawConstrainCoefficients = entity->rawConstrainCoefficients;
    }

    if (solver->constrainConstants.host()->size())
    {
      solver->constrainConstants.host()->insert(entity->constrainConstants.host()->end(),
        entity->constrainConstants.host()->begin(),
        solver->constrainConstants.host()->end());
    }
    else
    {
      *solver->constrainConstants.host() = *entity->constrainConstants.host();
    }

    uint entityId = getNewEntityId();
    vector<Real3>* entityPositions = entity->constrainConstants.host();
    for (uint p = 0; p < entityPositions->size(); p++)
    {
      ParticleStruct particle;
      particle.position = entityPositions->at(p);
      particle.identity = (i << PARTICLE_INSTANCE_ID_SHIFT) | entityId;
      solver->particles.host()->push_back(particle);
    }

    if (solver->particleAuxData.host()->size())
    {
      solver->particleAuxData.host()->insert(entity->particleAuxData.host()->end(),
        entity->particleAuxData.host()->begin(),
        solver->particleAuxData.host()->end());
    }
    else
    {
      *solver->particleAuxData.host() = *entity->particleAuxData.host();
    }

    if (solver->particleRigidData.host()->size())
    {
      solver->particleRigidData.host()->insert(entity->particleRigidData.host()->end(),
        entity->particleRigidData.host()->begin(),
        solver->particleRigidData.host()->end());
    }
    else
    {
      *solver->particleRigidData.host() = *entity->particleRigidData.host();
    }

    solver->commit();

    /*if (i == 0) // create entity header
    {
    ParticleSharedData* sharedData = &solver->particleSharedData.host()->back();
    sharedData->entityAlloc.offsets[DEVICE_HEADER_NODE] = solver->nodeOffset;
    sharedData->entityAlloc.offsets[DEVICE_HEADER_CONNECTION] = solver->connectionOffset;
    //sharedData->entityAlloc.counts[DEVICE_HEADER_NODE] = entity->nodeCount;
    //sharedData->entityAlloc.counts[DEVICE_HEADER_CONNECTION] = entity->connectionCount;
    entitySharedData.push_back(sharedData);
    }*/
    nodeCount += solver->nodeOffset;
  }

  updateInfo.counts[DEVICE_HEADER_NODE] = nodeCount - updateInfo.offsets[DEVICE_HEADER_NODE];
  updates.push_back(updateInfo);
}

void PhysicsSystem::step()
{
  if (updates.size())
  {
    SharedAllocator* allocator = allocators[0];

    for (const SectionData& section : updates)
    {
      // reset position delta for entity
      float zero = 0;

      compute->setBuffer(allocator->getHeap(COMPUTE_HEAP_PARTICLE_DELTA)->get(),
        section.offsets[DEVICE_HEADER_NODE] * sizeof(ParticleDifferential),
        section.counts[DEVICE_HEADER_NODE] * sizeof(ParticleDifferential),
        &zero, sizeof(float));
    }
    updates.clear();
  }
  step(.033f);
}

#ifdef ENABLE_RENDERING

const ParticleSharedData* PhysicsSystem::getEntitySharedData(PhysicsEntity* entity)const
{
  const vector<PhysicsEntity*>::const_iterator entityLocation = find(entities.begin(), entities.end(), entity);
  if (entityLocation != entities.end())
  {
    printf("Entity not found!");
    assert(0);
  }
  return entitySharedData[(entityLocation - entities.begin())];
}

void PhysicsSystem::render()
{
  compute->sync();
  for (uint i = 0; i < SOLVER_MAX; i++)
  {
    if (solversUint[i])
    {
      uint elements = solversUint[i]->nodes();
      solversUint[i]->particles.syncHost(0, elements);
    }
  }

  for (uint i = 0; i < entitySharedData.size(); i++)
  {
    uint entityOffset = entitySharedData[i]->entityAlloc.offsets[DEVICE_HEADER_NODE];
    entities[i]->render(&(*entityParticles[i])[entityOffset]);
  }
}

#endif

void PhysicsSystem::step(float timeStep)
{
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
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_DELTA)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_DIFF)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_SHARED)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_AUX)->get()
    };
    kernels[0].setArgs(buffers, 5);
    kernels[0].setArg<float>(&timeStep, 5);
    kernels[0].setArg<uint>(&nodeCount, 6);
    compute->execute(kernels[0], workgroupSize, workgroupCount);
  }
}