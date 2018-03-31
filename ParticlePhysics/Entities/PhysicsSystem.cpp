#include "PhysicsSystem.h"

#include "../Solvers/LinearSolver.h"
#include "../Solvers/DistanceSolver.h"
#include "../Solvers/RigidSolver.h"

PhysicsSystem::PhysicsSystem(ComputeInterface* compute)
  : compute(compute)
{
  nodeCount = 0;
  instanceNodeCount = 0;
  totalEntityCount = 0;
  availableEntityIds.clear();

  for (uint i = 0; i < SOLVER_MAX; i++)
  {
    solversUshort[i] = NULL;
    solversUint[i] = NULL;
  }

  includeFiles.push_back("ComputeHeader.shader");
  includeFiles.push_back("ComputeShared.h");
  includeFiles.push_back("ParticleStruct.h");

  vector<string> newType = { "uint", "float", "float3" };
  vector<string> oldType = { "IndexType", "CoefficientType", "VariableType" };
  registerShader(compute, "PhysicsSystem.shader", &oldType, &newType);
  kernels.push_back(programs[0].createKernel("integrate"));

  globalOffsets.create(compute, NULL, true);
  globalOffsets.resize(SOLVER_MAX, false);

  globalOffsets.host()->resize(SOLVER_MAX);

  for (uint i = 0; i < SOLVER_MAX; i++)
  {
    for (uint j = 0; j < 3; j++)
      (*globalOffsets.host())[i][j] = 0;
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

PhysicsEntityId PhysicsSystem::registerEntity(PhysicsEntity* entity)
{
  PhysicsEntityId entityId;

  // create memory heap allocators for the system
  if (!allocators.size())
  {
    SharedAllocator* allocator = new SharedAllocator(compute);
    int multiplier = 2048;
    allocator->particleAllocator.create(multiplier * 1024);
    allocator->constrainAllocator.create(multiplier * 1024, multiplier * 128);
    allocators.push_back(allocator);
  }

  Solver<uint, real, Real3>* solver = (Solver<uint, real, Real3>*)getSolver(entity->solver);

  // create section data to issue updates
  SectionData entitySectionData;
  SectionData systemUpdateInfo;

  entitySectionData.node.offset = solver->nodes();
  entitySectionData.connection.offset = solver->connectionCount();

  // add entity shared data to the system
  solver->entityParticleSharedData.host()->push_back(entity->entityParticleSharedData.host()->at(0));

  // update information
  systemUpdateInfo.node.offset = nodeCount;

  entityId.setSolverId(entity->solver, solver->newEntityId());

  // append data
  solver->rawConstrainConnections.insert(solver->rawConstrainConnections.end(),
    entity->rawConstrainConnections.begin(), entity->rawConstrainConnections.end());

  solver->rawConstrainCoefficients.insert(solver->rawConstrainCoefficients.end(),
    entity->rawConstrainCoefficients.begin(), entity->rawConstrainCoefficients.end());

  solver->constrainConstants.host()->insert(solver->constrainConstants.host()->end(),
    entity->constrainConstants.host()->begin(), entity->constrainConstants.host()->end());

  solver->particleAuxData.host()->insert(solver->particleAuxData.host()->end(),
    entity->particleAuxData.host()->begin(), entity->particleAuxData.host()->end());

  solver->particleRigidData.host()->insert(solver->particleRigidData.host()->end(),
    entity->particleRigidData.host()->begin(), entity->particleRigidData.host()->end());

  // increment total node count
  nodeCount += mMax(entity->constrainConstants.host()->size(), entity->particleRigidData.host()->size());

  solver->commit(entitySectionData);

  systemUpdateInfo.node.count = nodeCount - systemUpdateInfo.node.offset;

  entitySectionData.node.count = solver->nodes() - entitySectionData.node.offset;
  entitySectionData.connection.count = solver->connectionCount() - entitySectionData.connection.offset;

  // register entity properties
  entities.push_back(entity);
  this->entitySectionData.push_back(entitySectionData);
  updates.push_back(systemUpdateInfo);

  return entityId;
}

void PhysicsSystem::addEntityInstance(const PhysicsEntityId registeredEntityId, const ushort instanceCount, const Matrix4* instanceTransforms)
{
  // get entity
  const uint solverId = getSolverId(registeredEntityId);
  const PhysicsEntity* entity = entities[solverId];
  const vector<Real3>* entityPositions = entity->constrainConstants.host();
  Solver<uint, real, Real3>* solver = (Solver<uint, real, Real3>*)getSolver((SolverType)(1 << (getSolverType(registeredEntityId) - 1)));

  for (uint instance = 0; instance < instanceCount; instance++)
  {
    PhysicsEntityId entityInstanceId = registeredEntityId;
    entityInstanceId.setEntityId(solver->newEntityInstanceId());

    for (const Real3& position : *entityPositions)
    {
      ParticleStruct particle;
      instanceTransforms[instance].transformPos(particle.position, position);
      solver->particles.host()->push_back(particle);
      solver->particleIdentities.host()->push_back(entityInstanceId);
    }
    PartitionInfo partition;
    partition.offset = solver->lastPartition().end();
    partition.count = solver->deviceSections.host()->at(solverId).node.count;

    solver->partitions.host()->push_back(partition);
    instanceNodeCount += entityPositions->size();
  }

  uint cumulativeNode = 0;
  uint cumulativeSolver = 0;
  uint cumulativeInstance = 0;

  // update starting offset for each solver
  for (uint i = 2; i < SOLVER_MAX; i++)
  {
    Solver<uint, real, Real3>* localSolver = (Solver<uint, real, Real3>*)getSolver((SolverType)(1 << (i - 2)));
    if (localSolver)
    {
      cumulativeNode += localSolver->lastPartition().end();
      cumulativeSolver += localSolver->newEntityId();
      cumulativeInstance += localSolver->newEntityInstanceId();
    }
    (*globalOffsets.host())[i][GLOBAL_NODE_OFFSET] = cumulativeNode;
    (*globalOffsets.host())[i][GLOBAL_SOLVER_OFFSET] = cumulativeSolver;
    (*globalOffsets.host())[i][GLOBAL_INSTANCE_OFFSET] = cumulativeInstance;
  }

  globalOffsets.syncDevice();

  entityParticles.push_back(solver->particles.host());
  updates.push_back(SectionData());
}

void PhysicsSystem::step()
{
  glFinish();
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
        0, instanceNodeCount * sizeof(ParticleStruct), &zero, sizeof(float));
      compute->setBuffer(allocator->getHeap(COMPUTE_HEAP_PARTICLE_DIFF)->get(),
        0, instanceNodeCount * sizeof(ParticleDifferential), &zero, sizeof(float));
    }
    updates.clear();
  }
  step(.066f);

  compute->sync();
  ProfileManager::dumpAll(stdout);
  ProfileManager::Increment_Frame_Counter();
}

#ifdef ENABLE_RENDERING

void PhysicsSystem::render()
{
  for (uint i = 0; i < SOLVER_MAX; i++)
  {
    if (solversUint[i])
    {
      uint elements = solversUint[i]->lastPartition().end();
      if (!elements) continue;

      solversUint[i]->particles.syncHost(0, elements);

      for (const PartitionInfo &partition : *solversUint[i]->partitions.host())
      {
        IdentityInfo identity = solversUint[i]->particleIdentities.host()->at(partition.offset);
        uint solverType = getSolverType(identity);
        uint solverId = getSolverId(identity);
        uint entityOffset = partition.offset;

        entityOffset += globalOffsets.host()->at(solverType)[GLOBAL_INSTANCE_OFFSET];
        entities[solverId]->render(&((*(entityParticles[0]))[entityOffset]));
      }
    }
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
    compute->configureSize(workgroupSize, workgroupCount, instanceNodeCount);

    ComputeMemory* buffers[] = {
      allocator->getHeap(COMPUTE_HEAP_PARTICLE)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_IDENTITY)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_DELTA)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_DIFF)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_SHARED)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTICLE_AUX)->get(),
      allocator->getHeap(COMPUTE_HEAP_PARTITIONS)->get(),
      allocator->getHeap(COMPUTE_HEAP_SECTIONS)->get(),
      globalOffsets.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[0].setArgs(buffers, bufferCount);
    kernels[0].setArg<float>(&timeStep, bufferCount);
    kernels[0].setArg<uint>(&instanceNodeCount, bufferCount + 1);
    compute->execute(kernels[0], workgroupSize, workgroupCount);
  }
}