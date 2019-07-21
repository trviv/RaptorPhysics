#include "PhysicsSystem.h"

#include "../Solvers/LinearSolver.h"
#include "../Solvers/DistanceSolver.h"
#include "../Solvers/RigidSolver.h"

#include "../Solvers/Collision/UniformGridCollisionSolver.h"
#include "../Solvers/Collision/LBVHSolver.h"

//#define DEBUG_PHYSICS_SYSTEM

PhysicsSystem::PhysicsSystem(ComputeInterface* compute)
  : compute(compute)
{
  nodeCount = 0;
  instanceNodeCount = 0;
  availableEntityIds.clear();

  for (uint i = 0; i < SOLVER_MAX; i++)
  {
    solversUshort[i] = NULL;
    solversUint[i] = NULL;
#ifdef ENABLE_RENDERING
    solverParticleRadius[i].clear();
#endif
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

  globalOffsets.create(compute, NULL, true);
  globalOffsets.resize(SOLVER_MAX, false);

  globalOffsets.host()->resize(SOLVER_MAX);

  for (uint i = 0; i < SOLVER_MAX; i++)
  {
    (*globalOffsets.host())[i].globalNodeOffset = 0;
    (*globalOffsets.host())[i].globalInstanceOffset = 0;
    (*globalOffsets.host())[i].globalSolverOffset = 0;
  }

  indexMap.create(compute, NULL, true);

//  collisionSolver = new UniformGridCollisionSolver();
  collisionSolver = new LBVHSolver();

#ifdef ENABLE_RENDERING
  renderParticles = true;
  renderSolids = false;
#endif
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

void* PhysicsSystem::getSolver(SolverType type)
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

  // create memory heap allocators for the system
  if (!allocators.size())
  {
    SharedAllocator* allocator = new SharedAllocator(compute);
    int multiplier = 2048;
    allocator->particleAllocator.create(multiplier * 1024);
    allocator->constrainAllocator.create(multiplier * 1024, multiplier * 128);
    allocators.push_back(allocator);
    collisionSolver->init(compute, allocator);
  }

  Solver<uint, real, Real3>* solver = (Solver<uint, real, Real3>*)getSolver(entity->solver);

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

  solver->particleAuxData.host()->insert(solver->particleAuxData.host()->end(),
    entity->particleAuxData.host()->begin(), entity->particleAuxData.host()->end());

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
  Solver<uint, real, Real3>* solver = (Solver<uint, real, Real3>*)getSolver(solverType);

  const uint lastPartitionOffset = solver->entityLocations.host()->at(entityId).node.offset;

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
#ifdef ENABLE_RENDERING
      solverParticleRadius[solverType].push_back(solver->particleAuxData.host()->at(lastPartitionOffset + i).radius);
#endif
      solver->particles.host()->push_back(particle);
      solver->particleCollisionData.host()->push_back(entityParticleCol->at(i));
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
  for (uint i = 2; i < SOLVER_MAX; i++)
  {
    Solver<uint, real, Real3>* localSolver = (Solver<uint, real, Real3>*)getSolver((SolverType)(1 << (i - 2)));
    if (localSolver)
    {
      cumulativeNode += localSolver->lastPartition().end();
      cumulativeSolver += localSolver->newEntityId();
      cumulativeInstance += localSolver->newEntityInstanceId();
    }
    (*globalOffsets.host())[i].globalNodeOffset = cumulativeNode;
    (*globalOffsets.host())[i].globalSolverOffset = cumulativeSolver;
    (*globalOffsets.host())[i].globalInstanceOffset = cumulativeInstance;
  }

  globalOffsets.syncDevice();
  updates.push_back(EntityLocation());
}

void PhysicsSystem::step()
{
  glFinish();
  const float lastStepTime = ProfileManager::Get_Time_Since_Reset() / 1000.f;

  ProfileManager::Reset();

//  step(lastStepTime);
  step(1.f / 30.f);

  compute->sync();
  ProfileManager::dumpAll(stdout);
  ProfileManager::Increment_Frame_Counter();
}

#ifdef ENABLE_RENDERING

void PhysicsSystem::createSphere(float radius)
{
  vector<float> sphereVertices;
  vector<float> sphereNormals;
  vector<float> sphereTexcoords;
  vector<uint>  sphereIndices;

  uint rings = 12;
  uint sectors = 12;

  const float R = 1.f / (float)(rings - 1);
  const float S = 1.f / (float)(sectors - 1);
  uint r;
  uint s;

  sphereVertices.resize(rings * sectors * 3);
  sphereNormals.resize(rings * sectors * 3);
  sphereTexcoords.resize(rings * sectors * 2);
  vector<GLfloat>::iterator v = sphereVertices.begin();
  vector<GLfloat>::iterator n = sphereNormals.begin();
  vector<GLfloat>::iterator t = sphereTexcoords.begin();

  for (r = 0; r < rings; r++)
  {
    for (s = 0; s < sectors; s++)
    {
      const float y = sin(-M_PI_2 + M_PI * r * R);
      const float x = cos(2 * M_PI * s * S) * sin(M_PI * r * R);
      const float z = sin(2 * M_PI * s * S) * sin(M_PI * r * R);

      *t++ = s * S;
      *t++ = r * R;

      *v++ = x * radius;
      *v++ = y * radius;
      *v++ = z * radius;

      *n++ = x;
      *n++ = y;
      *n++ = z;
    }
  }

  sphereIndices.resize(rings * sectors * 6);
  vector<GLuint>::iterator i = sphereIndices.begin();
  for (r = 0; r < rings; r++)
  {
    for (s = 0; s < sectors; s++)
    {
      *i++ = r * sectors + s;
      *i++ = r * sectors + (s + 1);
      *i++ = (r + 1) * sectors + s;
      *i++ = (r + 1) * sectors + s;
      *i++ = r * sectors + (s + 1);
      *i++ = (r + 1) * sectors + (s + 1);
    }
  }
  displayVertex.copyData(&sphereVertices[0], rings * sectors, 0, 3 * sizeof(float));
  displayElements.copyData(&sphereIndices[0], sphereIndices.size());
}

void PhysicsSystem::createUnitBox()
{
  float boxVertices[] = {
    -1.0f, -1.0f, -1.0f,  +1.0f, -1.0f, -1.0f,  -1.0f, +1.0f, -1.0f,  +1.0f, +1.0f, -1.0f,
    -1.0f, -1.0f, +1.0f,  +1.0f, -1.0f, +1.0f,  -1.0f, +1.0f, +1.0f,  +1.0f, +1.0f, +1.0f,
  };

  displayBoxVertex.copyData(boxVertices, 8, 0, 3 * sizeof(float));

  uint boxIndices[] = {
    0, 1, 0, 2, 0, 4,
    1, 3, 1, 5,
    2, 3, 2, 6,
    3, 7,
    4, 5, 4, 6,
    5, 7,
    6, 7
  };
  displayBoxElements.copyData(boxIndices, 24);

  displayBoxVertex.bind();
  displayBoxShader.bindLocation(0, "position");
  GL_CHECK(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0));
  displayBoxVertex.unbind();
}

void PhysicsSystem::render()
{
  GL_CHECK(glEnable(GL_DEPTH_TEST));
  GL_CHECK(glDepthFunc(GL_LESS));
  GL_CHECK(glDisable(GL_BLEND));

  for (uint i = 0; i < SOLVER_MAX; i++)
  {
    if (solversUint[i])
    {
      uint elements = solversUint[i]->lastPartition().end();

      if (!elements) continue;

      solversUint[i]->particles.syncHost(0, elements);
      solversUint[i]->particleCollisionData.syncHost(0, elements);
      compute->sync();

      if (renderParticles)
      {
        // display particles
        ParticleStruct* particles = &((*solversUint[i]->particles.host())[0]);
        for (uint j = 0; j < elements; j++)
        {
          particles[j].radius = solverParticleRadius[i][j];
        }
        displayPositionBuffer.copy((float*)particles, 0, 0, elements);

        // display lines showing SDF data
        ParticleCollisionData* particleCol = &((*solversUint[i]->particleCollisionData.host())[0]);
        float* particleSdf = new float[elements * 4];
        for (uint j = 0; j < elements; j++)
        {
          particleSdf[j * 4] = particleCol[j].transformedSdfGradient.x;
          particleSdf[j * 4 + 1] = particleCol[j].transformedSdfGradient.y;
          particleSdf[j * 4 + 2] = particleCol[j].transformedSdfGradient.z;
          particleSdf[j * 4 + 3] = particleCol[j].radius;
        }
        displayAuxBuffer.copy((float*)particleSdf, 0, 0, elements);

        displayShader.bind();
        displayShader.set("modelViewMatrix", this->modelMatrix);
        displayShader.set("projectionMatrix", this->projectionMatrix);
        displayShader.activateTexture("particlePos", 0, displayPositionBuffer);
        displayShader.activateTexture("particleSDFGrad", 1, displayAuxBuffer);

        displayVertex.bind();
        displayElements.bind();
        GL_CHECK(glDrawElementsInstanced(GL_TRIANGLES, displayElements.count(), GL_UNSIGNED_INT, NULL, elements));
        displayElements.unbind();

        displayVertex.unbind();
        displayShader.unbind();

        displayLineShader.bind();
        displayLineVertex.bind();
        displayLineShader.set("modelViewMatrix", this->modelMatrix);
        displayLineShader.set("projectionMatrix", this->projectionMatrix);
        displayLineShader.activateTexture("particlePos", 0, displayPositionBuffer);
        displayLineShader.activateTexture("particleSDFGrad", 1, displayAuxBuffer);

        GL_CHECK(glDrawArraysInstanced(GL_LINES, 0, 2, elements));

        displayLineVertex.unbind();
        displayLineShader.unbind();
      }

      if (renderSolids)
      {
        // display solid
        for (const PartitionInfo &partition : *solversUint[i]->partitions.host())
        {
          IdentityInfo identity = solversUint[i]->particles.host()->at(partition.offset).identity;
          uint solverId = getEntityId(identity);
          ParticleStruct* pos = &((*solversUint[i]->particles.host())[partition.offset]);

          entities[i][solverId]->render(pos);
        }
      }
    }
  }

  // render boundign boxes if supplied by the colision solver
  if (false && collisionSolver->getBoundingBoxes())
  {
    DeviceArray<XAB>* collisionBoundingBoxes = collisionSolver->getBoundingBoxes();
    collisionBoundingBoxes->syncHost();
    compute->sync();

    XAB* boxes = &((*collisionBoundingBoxes->host())[0]);

    displayBoxBuffer.copy((float*)boxes, 0, 0, collisionBoundingBoxes->host()->size() * 2);

    GL_CHECK(glEnable(GL_BLEND));
    GL_CHECK(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

    displayBoxShader.bind();
    displayBoxShader.set("modelViewMatrix", this->modelMatrix);
    displayBoxShader.set("projectionMatrix", this->projectionMatrix);
    displayBoxShader.activateTexture("boundingBoxes", 0, displayBoxBuffer);

    displayBoxVertex.bind();
    displayBoxElements.bind();
    GL_CHECK(glDrawElementsInstanced(GL_LINES, displayBoxElements.count(), GL_UNSIGNED_INT, NULL, collisionBoundingBoxes->host()->size()));
    displayBoxElements.unbind();
    displayBoxVertex.unbind();

    displayBoxShader.unbind();
  }
}

#endif

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
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_DELTA)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_DIFF)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_SHARED)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_AUX)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTITIONS)->get(),
    allocator->getHeap(COMPUTE_HEAP_SECTIONS)->get(),
    globalOffsets.device()
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
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_DELTA)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_DIFF)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_SHARED)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_AUX)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTITIONS)->get(),
    allocator->getHeap(COMPUTE_HEAP_SECTIONS)->get(),
    globalOffsets.device()
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

  uint firstStep = !updates.size();

  if (updates.size())
  {
    ProfileBlock("Physics system update");
    SharedAllocator* allocator = allocators[0];

    // reset position delta for entity
    ParticleStruct dummy;
    dummy.position.x = 0.f;
    dummy.position.y = 0.f;
    dummy.position.z = 0.f;
    dummy.identity.identity = 0;

    ComputeUtil::get(0)->clearIntegerBuffer(compute, allocator->getHeap(COMPUTE_HEAP_PARTICLE_DELTA)->get(),
      instanceNodeCount * sizeof(ParticleStruct)/sizeof(uint));
    ComputeUtil::get(0)->clearIntegerBuffer(compute, allocator->getHeap(COMPUTE_HEAP_PARTICLE_DIFF)->get(),
      instanceNodeCount * sizeof(ParticleStruct)/sizeof(uint));

    updates.clear();

#ifdef ENABLE_RENDERING
    uint width = 16;
    uint height = (instanceNodeCount + 15) / 16;
    displayVertex.gen();
    displayLineVertex.gen();
    displayElements.gen();
    displayBoxVertex.gen();
    displayBoxElements.gen();

    displayPositionBuffer.init(width, height);
    displayPositionBuffer.gen();
    displayColorBuffer.init(width, height);
    displayColorBuffer.gen();
    displayAuxBuffer.init(width, height);
    displayAuxBuffer.gen();
    displayBoxBuffer.init(128, (instanceNodeCount * 2 + 127) / 128);
    displayBoxBuffer.gen();

    clearColor[0] = 0.7f;
    clearColor[1] = 0.7f;
    clearColor[2] = 0.7f;
    clearColor[3] = 1.0f;

    if (renderParticles)
    {
      displayShader.init("ParticleVert.glsl", "ParticleFrag.glsl");
      displayLineShader.init("LineVert.glsl", "LineFrag.glsl");

      createSphere(1.f);
      float line[] = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };
      displayLineVertex.copyData(line, 2, 0, 3 * sizeof(float));

      displayVertex.bind();
      displayShader.bindLocation(0, "position");
      GL_CHECK(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0));
      displayVertex.unbind();

      displayShader.linkPrograms();
      displayLineShader.linkPrograms();
    }
    else if (renderSolids)
    {
      displayShader.init("SolidVert.glsl", "SolidFrag.glsl");
      displayLineShader.init("LineVert.glsl", "LineFrag.glsl");

      displayShader.linkPrograms();
      displayLineShader.linkPrograms();
    }

    displayBoxShader.init("BoxVert.glsl", "BoxFrag.glsl");
    createUnitBox();
    displayBoxShader.linkPrograms();
#endif

    indexMap.resize(instanceNodeCount, false);
  }

//  if (firstStep)
//  {
//    positionUpdate(timeStep);
//  }
//  else
  {
    integrate(timeStep);
  }

  collisionSolver->solve(instanceNodeCount, globalOffsets.device());

  for (uint i = 0; i < SOLVER_MAX; i++)
  {
    if (solversUint[i])
    {
      solversUint[i]->solve();
    }
  }

  differentiate(timeStep);
}
