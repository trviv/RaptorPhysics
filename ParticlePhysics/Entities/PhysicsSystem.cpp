#include "PhysicsSystem.h"

#include "../Solvers/LinearSolver.h"
#include "../Solvers/DistanceSolver.h"
#include "../Solvers/RigidSolver.h"
#include "../Solvers/FluidSolver.h"
#include "../Solvers/FluidSolverPBF.h"
#include "../Solvers/FluidSolverPCISPH.h"

#include "../Solvers/Collision/UniformGridCollisionSolver.h"
#include "../Solvers/Collision/LBVHSolver.h"

//#define DEBUG_PHYSICS_SYSTEM
#define PHYSICS_SYSTEM_SINGLE_UPDATE

void PhysicsSystem::init(ComputeInterface* compute, const uint maxParticles)
{
  this->compute = compute;
  nodeCount = 0;
  instanceNodeCount = 0;
  availableEntityIds.clear();
  allocators.clear();
  updates.clear();

  for (uint i = 0; i < SOLVER_MAX; i++)
  {
    entities[i].clear();
    solversUshort[i] = NULL;
    solversUint[i] = NULL;
  }

  includeFiles.push_back("ComputeHeader.shader");
  includeFiles.push_back("ComputeShared.h");
  includeFiles.push_back("ParticleStruct.h");

  vector<string> newType = { "uint", "float", "float3" };
  vector<string> oldType = { "IndexType", "CoefficientType", "VariableType" };
  registerShader(compute, "PhysicsSystem.shader", &oldType, &newType);
  kernels.push_back(programs[0].createKernel("startStep"));
  kernels.push_back(programs[0].createKernel("endStep"));
  kernels.push_back(programs[0].createKernel("integrateDifferentiateStep"));

  systemSettings.create(compute, NULL);
  systemSettings.resize(1, false);
  systemSettings.host()->resize(1);
  systemSettings.host()->at(0).systemBound.min = Real3(-20.f, 0.f, -20.f);
  systemSettings.host()->at(0).systemBound.max = Real3(20.f, 0.f, 20.f);
  systemSettings.host()->at(0).gravity = Real3(0.f, -9.8f, 0.f);

  for (uint i = 0; i < SOLVER_MAX; i++)
  {
    systemSettings.host()->at(0).globalOffsets[i].globalNodeOffset = 0;
    systemSettings.host()->at(0).globalOffsets[i].globalInstanceOffset = 0;
    systemSettings.host()->at(0).globalOffsets[i].globalSolverOffset = 0;
  }

  systemSettings.syncDevice();

  indexMap.create(compute, NULL);

  // create memory heap allocators for the system
  if (!allocators.size())
  {
    SharedAllocator* allocator = new SharedAllocator(compute);
    allocator->particleAllocator.create(maxParticles);
    allocator->constrainAllocator.create(maxParticles, maxParticles);
    allocators.push_back(allocator);
    collisionSolver = new UniformGridCollisionSolver(compute, allocator);
//    collisionSolver = new LBVHSolver(compute, allocator);
    collisionSolver->init();
  }

  simulationIterations = 1;
  solverIterations = 1;
}

PhysicsSystem::~PhysicsSystem()
{
  for (uint s = 0; s < SOLVER_MAX; s++)
  {
    for (uint i = 0; entities[s].size() && i < entities[s].size(); i++)
    {
      delete entities[s][i];
      entities[s][i] = NULL;
    }
  }

  delete collisionSolver;
}

uint PhysicsSystem::particleCount()const
{
  return instanceNodeCount;
}

const CollisionSolver* PhysicsSystem::getCollisionSolver()const
{
  return collisionSolver;
}

const PhySystemSettings& PhysicsSystem::getSystemSettings()const
{
  return systemSettings.host()->at(0);
}

vector<PhysicsEntity*>& PhysicsSystem::getEntities(SolverType type)
{
  return entities[type];
}

const EntitySolverType* PhysicsSystem::getSolver(SolverType type)const
{
  return solversUint[type];
}

EntitySolverType* PhysicsSystem::getAndInitSolver(SolverType type)
{
  // get solver
  int index = type;

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
    case SOLVER_FLUID:
    {
//      solversUint[index] = new FluidSolver(compute, allocators[0]);
//      solversUint[index] = new FluidSolverPBF(compute, allocators[0]);
      solversUint[index] = new FluidSolverPCISPH(compute, allocators[0]);
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
  resetIdentity(entityId);

  EntitySolverType* solver = getAndInitSolver(entity->solver);

  // create section data to issue updates
  EntityLocation systemUpdateInfo;

  // add entity shared data to the system
  solver->entitySharedData.host()->push_back(entity->entitySharedData.host()->at(0));

  // update information
  systemUpdateInfo.node.offset = nodeCount;

  setEntityId(entityId, entity->solver, solver->newEntityId());

  // append data
  solver->rawConstrainConnections.insert(solver->rawConstrainConnections.end(),
    entity->rawConstrainConnections.begin(), entity->rawConstrainConnections.end());

  solver->rawConstrainCoefficients.insert(solver->rawConstrainCoefficients.end(),
    entity->rawConstrainCoefficients.begin(), entity->rawConstrainCoefficients.end());

  solver->constrainConstants.host()->insert(solver->constrainConstants.host()->end(),
    entity->constrainConstants.host()->begin(), entity->constrainConstants.host()->end());

  solver->particleRigidData.host()->insert(solver->particleRigidData.host()->end(),
    entity->particleRigidData.host()->begin(), entity->particleRigidData.host()->end());

  // increment total node count
  nodeCount += mMax(entity->constrainConstants.host()->size(), entity->particleRigidData.host()->size());

  solver->commit();

  systemUpdateInfo.node.count = nodeCount - systemUpdateInfo.node.offset;

  // register entity properties
  entities[entity->solver].push_back(entity);
  updates.push_back(systemUpdateInfo);

  return entityId;
}

void PhysicsSystem::addEntityInstance(const PhysicsEntityId registeredEntityId, const ushort instanceCount, const Matrix4* instanceTransforms)
{
  // get entity
  const uint entityId = getEntityId(registeredEntityId);
  const SolverType solverType = (SolverType)getSolverType(registeredEntityId);
  const PhysicsEntity* entity = entities[getSolverType(registeredEntityId)][entityId];
  const vector<Real3>* entityPositions = entity->constrainConstants.host();
  const vector<ParticleCollisionData>* entityParticleCol = entity->particleCollisionData.host();
  EntitySolverType* solver = getAndInitSolver(solverType);

  for (uint instance = 0; instance < instanceCount; instance++)
  {
    PhysicsEntityId entityInstanceId = registeredEntityId;
    setInstanceId(entityInstanceId, solver->newEntityInstanceId());

    for (uint i = 0; i < entityPositions->size(); i++)
    {
      ParticleStruct particle;
      Real3 pos;
      instanceTransforms[instance].transformPos(pos, entityPositions->at(i));
      particle.position = pos;
      particle.identity = entityInstanceId;
      solver->particles.host()->push_back(particle);
      solver->particleCollisionData.host()->push_back(entityParticleCol->at(i));
      solver->particleCouplingData.host()->push_back(ParticleCouplingData());
    }

    PartitionInfo partition;
    partition.offset = solver->lastPartition().end();
    partition.count = solver->entityLocations.host()->at(entityId).node.count;

    solver->partitions.host()->push_back(partition);
    instanceNodeCount += entityPositions->size();
  }

  uint cumulativeNode = 0;
  uint cumulativeSolver = 0;
  uint cumulativeInstance = 0;

  // update starting offset for each solver
  for (uint i = 1; i < SOLVER_MAX-1; i++)
  {
    if (solversUint[i])
    {
      EntitySolverType* localSolver = getAndInitSolver((SolverType)i);
      cumulativeNode += localSolver->lastPartition().end();
      cumulativeSolver += localSolver->newEntityId();
      cumulativeInstance += localSolver->newEntityInstanceId();
    }
    systemSettings.host()->at(0).globalOffsets[i+1].globalNodeOffset = cumulativeNode;
    systemSettings.host()->at(0).globalOffsets[i+1].globalSolverOffset = cumulativeSolver;
    systemSettings.host()->at(0).globalOffsets[i+1].globalInstanceOffset = cumulativeInstance;
  }
  systemSettings.syncDevice();

  updates.push_back(EntityLocation());
}

void PhysicsSystem::step()
{
  for (int i=0; i<simulationIterations; i++)
  {
    step(1.f / (simulationIterations * 60));
  }

  compute->sync(false);
}

void PhysicsSystem::integrate(float timeStep)
{
  SharedAllocator* allocator = allocators[0];

  size_t workgroupSize[3], workgroupCount[3];
  compute->configureSize(workgroupSize, workgroupCount, instanceNodeCount);

  ComputeMemory* buffers[] = {
    allocator->getHeap(COMPUTE_HEAP_PARTICLE)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_PREDICTED)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_DIFF)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_SHARED)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_COLLISION)->get(),
    systemSettings.device()
  };
  uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
  kernels[0].setArgs(buffers, bufferCount);
  kernels[0].setArg<float>(&timeStep, bufferCount);
  kernels[0].setArg<uint>(&instanceNodeCount, bufferCount + 1);
  compute->execute(kernels[0], workgroupSize, workgroupCount);

#ifdef DEBUG_PHYSICS_SYSTEM
  compute->sync();
#endif
}

void PhysicsSystem::differentiate(float timeStep)
{
  SharedAllocator* allocator = allocators[0];

  size_t workgroupSize[3], workgroupCount[3];
  compute->configureSize(workgroupSize, workgroupCount, instanceNodeCount);

  ComputeMemory* buffers[] = {
    allocator->getHeap(COMPUTE_HEAP_PARTICLE)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_PREDICTED)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_DIFF)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_FORCE)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_SHARED)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_COLLISION)->get(),
    systemSettings.device()
  };
  uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
  kernels[1].setArgs(buffers, bufferCount);
  kernels[1].setArg<float>(&timeStep, bufferCount);
  kernels[1].setArg<uint>(&instanceNodeCount, bufferCount + 1);
  compute->execute(kernels[1], workgroupSize, workgroupCount);

#ifdef DEBUG_PHYSICS_SYSTEM
  compute->sync();
#endif
}

void PhysicsSystem::positionUpdate(float timeStep)
{
  SharedAllocator* allocator = allocators[0];

  size_t workgroupSize[3], workgroupCount[3];
  compute->configureSize(workgroupSize, workgroupCount, instanceNodeCount);

  ComputeMemory* buffers[] = {
    allocator->getHeap(COMPUTE_HEAP_PARTICLE)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_PREDICTED)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_DIFF)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_FORCE)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_SHARED)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_COLLISION)->get(),
    systemSettings.device()
  };
  uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
  kernels[2].setArgs(buffers, bufferCount);
  kernels[2].setArg<float>(&timeStep, bufferCount);
  kernels[2].setArg<uint>(&instanceNodeCount, bufferCount + 1);
  compute->execute(kernels[2], workgroupSize, workgroupCount);

#ifdef DEBUG_PHYSICS_SYSTEM
  compute->sync();
#endif
}

void PhysicsSystem::step(float timeStep)
{
  ProfileBlock("Physics system step");

  for (uint i = 0; i < SOLVER_MAX; i++)
  {
    if (solversUint[i])
    {
      solversUint[i]->update();
    }
  }

  uint firstStep = (uint)updates.size();

  if (updates.size())
  {
    ProfileBlock("Physics system update");
    SharedAllocator* allocator = allocators[0];

    ComputeUtil::get(0)->clearBuffer(compute, allocator->getHeap(COMPUTE_HEAP_PARTICLE_DIFF)->get(), instanceNodeCount * sizeof(ParticleDifferential)/sizeof(uint), 0);
    ComputeUtil::get(0)->clearBuffer(compute, allocator->getHeap(COMPUTE_HEAP_PARTICLE_FORCE)->get(), instanceNodeCount * sizeof(ParticleForce)/sizeof(uint), 0);

    updates.clear();

    indexMap.resize(instanceNodeCount, false);

    // set system data pointers in fluid solver
    if (solversUint[SOLVER_FLUID] != NULL)
    {
      // TODO: Probably move it to a place less frequently updated
      ((FluidSolver*)solversUint[SOLVER_FLUID])->systemParticlePositions      = allocators[0]->getHeap(COMPUTE_HEAP_PARTICLE_PREDICTED)->get();
      ((FluidSolver*)solversUint[SOLVER_FLUID])->systemParticleDifferential   = allocators[0]->getHeap(COMPUTE_HEAP_PARTICLE_DIFF)->get();
      ((FluidSolver*)solversUint[SOLVER_FLUID])->systemParticleCouplingData   = allocators[0]->getHeap(COMPUTE_HEAP_PARTICLE_COUPLING)->get();
      ((FluidSolver*)solversUint[SOLVER_FLUID])->systemParticleForce          = allocators[0]->getHeap(COMPUTE_HEAP_PARTICLE_FORCE)->get();
      ((FluidSolver*)solversUint[SOLVER_FLUID])->systemParticleCount          = instanceNodeCount;
      ((FluidSolver*)solversUint[SOLVER_FLUID])->systemSettings               = systemSettings.device();
    }

    // create coupling data for rigid bodies
    if (solversUint[SOLVER_RIGID_BODY] && solversUint[SOLVER_FLUID])
    {
      RigidSolver* rigidSolver = (RigidSolver*)solversUint[SOLVER_RIGID_BODY];

      rigidSolver->particles.syncHost();
      compute->sync();
      const vector<ParticleStruct>& hostParticles = *(rigidSolver->particles.host());

      DeviceArray<ParticleStruct> particles(compute);
      DeviceArray<ParticleCouplingData> particleCouplingData(compute, NULL);
      DeviceArray<ParticleCollisionData> particleCollisionData(compute);

      for (int entityId=0; entityId<rigidSolver->entityLocations.host()->size(); entityId++)
      {
        const PartitionInfo &partition = rigidSolver->entityLocations.host()->at(entityId).node;
        if (particles.size() < partition.count)
        {
          particles.resize(partition.count, false);
          particleCouplingData.resize(partition.count, false);
          particleCollisionData.resize(partition.count, false);
        }
        compute->copyBuffer(rigidSolver->particles.device(), particles.device(), partition.offset * sizeof(ParticleStruct), 0, sizeof(ParticleStruct) * partition.count);
        compute->copyBuffer(rigidSolver->particleCollisionData.device(), particleCollisionData.device(), partition.offset * sizeof(ParticleCollisionData), 0, sizeof(ParticleCollisionData) * partition.count);

        ((FluidSolver*)solversUint[SOLVER_FLUID])->calculateParticleCouplingData(particleCouplingData, particles, particleCollisionData, partition.count);
#ifdef DEBUG_PHYSICS_SYSTEM
        particleCouplingData.syncHost();
        compute->sync();
#endif
        // loop over partitions and copy the calculated values to instances of this entity
        for (const PartitionInfo &partition : *(rigidSolver->partitions.host()))
        {
          if (getEntityId(hostParticles[partition.offset].identity) == entityId)
          {
            compute->copyBuffer(particleCouplingData.device(), rigidSolver->particleCouplingData.device(), 0, partition.offset * sizeof(ParticleCouplingData), sizeof(ParticleCouplingData) * partition.count);
          }
        }
      }
    }
  }

  // create coupling data for cloth
  if (solversUint[SOLVER_CLOTH] && solversUint[SOLVER_FLUID])
  {
    DistanceSolver* clothSolver = (DistanceSolver*)solversUint[SOLVER_CLOTH];
    ((FluidSolver*)solversUint[SOLVER_FLUID])->calculateParticleCouplingData(clothSolver->particleCouplingData, clothSolver->particles, clothSolver->particleCollisionData, clothSolver->particles.size());
  }

#ifdef PHYSICS_SYSTEM_SINGLE_UPDATE
  if (!firstStep)
  {
    positionUpdate(timeStep);
  }
  else
#endif
  {
    integrate(timeStep);
  }

  for (uint si = 0; si < solverIterations; si++)
  {
    for (uint i = 0; i < SOLVER_MAX; i++)
    {
      if (solversUint[i])
      {
        solversUint[i]->solve(timeStep / solverIterations);
      }
    }

    collisionSolver->solve(instanceNodeCount, systemSettings.device());
  }

  for (uint i = 0; i < SOLVER_MAX; i++)
  {
    if (solversUint[i])
    {
      solversUint[i]->postCollisionSolve(timeStep);
    }
  }

#ifndef PHYSICS_SYSTEM_SINGLE_UPDATE
  differentiate(timeStep);
#endif
}

void PhysicsSystem::setGravity(const Real3& gravity)
{
  systemSettings.host()->at(0).gravity = gravity;
  systemSettings.syncDevice();
}

void PhysicsSystem::setSystemBoundary(const XAB& bound)
{
  systemSettings.host()->at(0).systemBound = bound;
  systemSettings.syncDevice();
}
