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

static const string RENDER_PARTICLES_OPTION       ("Particles");
static const string RENDER_SOLIDS_OPTION          ("Solids");
static const string RENDER_BOUNDING_BOXES_OPTION  ("Bounding Boxes");
static const string RENDER_SYSTEM_BOUND_OPTION    ("Scene Box");
static const string RENDER_GRID_HEATMAP_OPTION    ("Grid Heatmap");
static const string PAUSE_SIM_OPTION              ("Pause Sim");
static const string RENDER_RESET_CAMERA_OPTION    ("Reset Camera");

static Clock physicsSystemClock;

void PhysicsSystem::init(ComputeInterface* compute, const uint maxParticles)
{
  this->compute = compute;
  displayGridBuffer = Texture(TEXTURE_FORMAT_INT);
  nodeCount = 0;
  instanceNodeCount = 0;
  availableEntityIds.clear();
  allocators.clear();
  updates.clear();
  elapsedSimTime = 0.f;
  frameCount = 0;

  for (uint i = 0; i < SOLVER_MAX; i++)
  {
    entities[i].clear();
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

  systemSettings.create(compute, NULL, true);
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

  indexMap.create(compute, NULL, true);

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

  addFrameOption(UIElement(RENDER_PARTICLES_OPTION, true, "fa-solid-900", 0xF141));
  addFrameOption(UIElement(RENDER_SOLIDS_OPTION, true, "fa-solid-900", 0xF1B3));
  addFrameOption(UIElement(RENDER_BOUNDING_BOXES_OPTION, true, "fa-brands-400", 0xF247));
  addFrameOption(UIElement(RENDER_SYSTEM_BOUND_OPTION, true, "fa-brands-400", 0xF1CB));
  addFrameOption(UIElement(RENDER_GRID_HEATMAP_OPTION, true, "fa-solid-900", 0xF37F));
  addFrameOption(UIElement(PAUSE_SIM_OPTION, false, "fa-solid-900", 0xF04C));
  addFrameOption(UIElement(RENDER_RESET_CAMERA_OPTION, RENDER_RESET_CAMERA_OPTION, "fa-solid-900", 0xF03D));

  bindParameter("renderParticlesOption", &getFrameOption(RENDER_PARTICLES_OPTION).boolValue, InputParameterType::ParameterTypeBool);
  bindParameter("renderSolidsOption", &getFrameOption(RENDER_SOLIDS_OPTION).boolValue, InputParameterType::ParameterTypeBool);
  bindParameter("renderBoundingBoxesOption", &getFrameOption(RENDER_BOUNDING_BOXES_OPTION).boolValue, InputParameterType::ParameterTypeBool);
  bindParameter("renderSystemBoundOption", &getFrameOption(RENDER_SYSTEM_BOUND_OPTION).boolValue, InputParameterType::ParameterTypeBool);
  bindParameter("renderGridHeatmapOption", &getFrameOption(RENDER_GRID_HEATMAP_OPTION).boolValue, InputParameterType::ParameterTypeBool);

#ifdef ENABLE_RENDERING
  elapsedRenderTime = 0.f;
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
    case SOLVER_FLUID:
    {
      solversUint[index] = new FluidSolver(compute, allocators[0]);
//      solversUint[index] = new FluidSolverPBF(compute, allocators[0]);
//      solversUint[index] = new FluidSolverPCISPH(compute, allocators[0]);
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

  EntitySolver<uint, real, Real3>* solver = (EntitySolver<uint, real, Real3>*)getSolver(entity->solver);

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
  EntitySolver<uint, real, Real3>* solver = (EntitySolver<uint, real, Real3>*)getSolver(solverType);

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
    EntitySolver<uint, real, Real3>* localSolver = (EntitySolver<uint, real, Real3>*)getSolver((SolverType)(1 << (i - 2)));
    if (localSolver)
    {
      cumulativeNode += localSolver->lastPartition().end();
      cumulativeSolver += localSolver->newEntityId();
      cumulativeInstance += localSolver->newEntityInstanceId();
    }
    systemSettings.host()->at(0).globalOffsets[i].globalNodeOffset = cumulativeNode;
    systemSettings.host()->at(0).globalOffsets[i].globalSolverOffset = cumulativeSolver;
    systemSettings.host()->at(0).globalOffsets[i].globalInstanceOffset = cumulativeInstance;
  }
  systemSettings.syncDevice();

  updates.push_back(EntityLocation());
}

void PhysicsSystem::step()
{
  // record render time
  const float renderTime = physicsSystemClock.getTimeMilliseconds();
  physicsSystemClock.reset();

  step(1.f / 60.f);

#ifdef ENABLE_RENDERING
  // sync all output buffers
  for (uint solver = 0; solver < SOLVER_MAX; solver++)
  {
    if (solversUint[solver])
    {
      uint elements = solversUint[solver]->lastPartition().end();

      if (!elements) continue;

      solversUint[solver]->particles.syncHost(0, elements);
      solversUint[solver]->particleCollisionData.syncHost(0, elements);
    }
  }

  if (getFrameOption(RENDER_BOUNDING_BOXES_OPTION).boolValue && collisionSolver->particleGroupBoundingBoxes.size())
  {
    collisionSolver->particleGroupBoundingBoxes.syncHost();
  }

  if (getFrameOption(RENDER_SYSTEM_BOUND_OPTION).boolValue && collisionSolver->systemBoundingBox.size())
  {
    collisionSolver->systemBoundingBox.syncHost();
  }

  if (getFrameOption(RENDER_GRID_HEATMAP_OPTION).boolValue && ((UniformGridCollisionSolver*)collisionSolver)->gridCellParticleCount.size())
  {
    ((UniformGridCollisionSolver*)collisionSolver)->gridCellParticleCount.syncHost();
    collisionSolver->systemBoundingBox.syncHost();
  }
#endif

  compute->sync(false);

  elapsedSimTime += physicsSystemClock.getTimeMilliseconds();
  elapsedRenderTime += renderTime;

#ifdef ENABLE_RENDERING

  // Display frame info
#define GUI_REFRESH_AFTER_FRAMES 0xF
  if ((frameCount & GUI_REFRESH_AFTER_FRAMES) == 0)
  {
    frameText.clear();

    char temp[64];
    sprintf(temp, "Particles:   %d\n", instanceNodeCount);
    frameText += temp;

    uint vertexCount = 0;
    if (getFrameOption(RENDER_PARTICLES_OPTION).boolValue)
    {
      vertexCount += displayParticleVertex.count() * instanceNodeCount;
      vertexCount += displayLineVertex.count() * instanceNodeCount;
    }
    if (getFrameOption(RENDER_SOLIDS_OPTION).boolValue)
    {
      // TODO: Find a good way to find this value, ignore for now
      //vertexCount += displaySolidVertex.count();
    }
    if (getFrameOption(RENDER_BOUNDING_BOXES_OPTION).boolValue && collisionSolver->particleGroupBoundingBoxes.size())
    {
      vertexCount += displayBoxVertex.count() * collisionSolver->particleGroupBoundingBoxes.host()->size();
    }
    if (getFrameOption(RENDER_SYSTEM_BOUND_OPTION).boolValue && collisionSolver->systemBoundingBox.size())
    {
      vertexCount += displayBoxVertex.count() * collisionSolver->systemBoundingBox.host()->size();
    }
    if (getFrameOption(RENDER_GRID_HEATMAP_OPTION).boolValue && ((UniformGridCollisionSolver*)collisionSolver)->gridCellParticleCount.size())
    {
      vertexCount += displayBoxVertex.count() * ((UniformGridCollisionSolver*)collisionSolver)->gridCellParticleCount.size();
    }
    sprintf(temp, "Vertices:    %d\n", vertexCount);
    frameText += temp;
    sprintf(temp, "Sim Time:    %.1f ms\n", elapsedSimTime / GUI_REFRESH_AFTER_FRAMES);
    frameText += temp;
    sprintf(temp, "Render Time: %.1f ms\n", elapsedRenderTime / GUI_REFRESH_AFTER_FRAMES);
    frameText += temp;
    elapsedSimTime = 0.f;
    elapsedRenderTime = 0.f;
  }
  frameCount++;
#endif

  physicsSystemClock.reset();
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
  displayParticleVertex.copyData(&sphereVertices[0], rings * sectors, 0, 3 * sizeof(float));
  displayParticleElements.copyData(&sphereIndices[0], (uint)sphereIndices.size());

  displayParticleVertex.bind();
  GL_CHECK(glEnableVertexAttribArray(0));
  GL_CHECK(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0));
  displayPositionBuffer.bind();
  GL_CHECK(glEnableVertexAttribArray(1));
  GL_CHECK(glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0));
  GL_CHECK(glVertexAttribDivisor(1, 1));
  displayCollisionBuffer.bind();
  GL_CHECK(glEnableVertexAttribArray(2));
  GL_CHECK(glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0));
  GL_CHECK(glVertexAttribDivisor(2, 1));
  displayParticleVertex.unbind();
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
  GL_CHECK(glEnableVertexAttribArray(0));
  GL_CHECK(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0));
  displayBoxVertex.unbind();

  uint gridIndices[] = {
    0, 1, 2, 2, 1, 3,
    0, 6, 4, 6, 0, 2,
    0, 4, 1, 1, 4, 5
  };
  displayGridElements.copyData(gridIndices, sizeof(gridIndices) / sizeof(uint));
}

void PhysicsSystem::createUnitCircle()
{
  const uint triangles = 8;

  vector<float> circleVertex;
  circleVertex.reserve(2 + triangles);

  circleVertex.push_back(0.f);
  circleVertex.push_back(0.f);
  circleVertex.push_back(0.f);

  for(int i = 0; i<=triangles; i++)
  {
    circleVertex.push_back(cos(i * 2.0f * M_PI / triangles));
    circleVertex.push_back(sin(i * 2.0f * M_PI / triangles));
    circleVertex.push_back(0.f);
  }

  displayFlatVertex.copyData(&circleVertex[0], 2 + triangles, 0, 3 * sizeof(float));

  displayFlatVertex.bind();
  GL_CHECK(glEnableVertexAttribArray(0));
  GL_CHECK(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0));
  displayPositionBuffer.bind();
  GL_CHECK(glEnableVertexAttribArray(1));
  GL_CHECK(glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0));
  GL_CHECK(glVertexAttribDivisor(1, 1));
  displayCollisionBuffer.bind();
  GL_CHECK(glEnableVertexAttribArray(2));
  GL_CHECK(glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0));
  GL_CHECK(glVertexAttribDivisor(2, 1));
  displayFlatVertex.unbind();
}

void PhysicsSystem::render()
{
  displayParticleShader.bind();
  displayParticleShader.set("modelViewMatrix", this->modelMatrix);
  displayParticleShader.set("projectionMatrix", this->projectionMatrix);
  displayParticleShader.unbind();

  displaySolidShader.bind();
  displaySolidShader.set("modelViewMatrix", this->modelMatrix);
  displaySolidShader.set("projectionMatrix", this->projectionMatrix);
  displaySolidShader.unbind();

  displayFlatShader.bind();
  displayFlatShader.set("modelViewMatrix", this->modelMatrix);
  displayFlatShader.set("projectionMatrix", this->projectionMatrix);
  displayFlatShader.unbind();

  displayBoxShader.bind();
  displayBoxShader.set("modelViewMatrix", this->modelMatrix);
  displayBoxShader.set("projectionMatrix", this->projectionMatrix);
  displayBoxShader.unbind();

  displayLineShader.bind();
  displayLineShader.set("modelViewMatrix", this->modelMatrix);
  displayLineShader.set("projectionMatrix", this->projectionMatrix);
  displayLineShader.unbind();

  displayGridShader.bind();
  displayGridShader.set("modelViewMatrix", this->modelMatrix);
  displayGridShader.set("projectionMatrix", this->projectionMatrix);
  displayGridShader.unbind();

  if (cameraInterface && cameraInterface->isActive())
  {
    if (displayBackgroundBuffer.getComputeTexture() == NULL)
    {
      displayBackgroundBuffer = createSharedTexture(compute, (uint[2]){cameraInterface->width(), cameraInterface->height()}, SHARED_TEXTURE_FORMAT_UINT8x4);
      displayBackgroundShader.bind();
      displayBackgroundShader.set("frameDimensions", (float)width(), (float)height(),
                                  (float)displayBackgroundBuffer.getGraphicsTexture().width(),
                                  (float)displayBackgroundBuffer.getGraphicsTexture().height());
      displayBackgroundShader.unbind();
    }

    if (cameraInterface->getCurrentFrame())
    {
      GL_CHECK(glDisable(GL_DEPTH_TEST));
      GL_CHECK(glDisable(GL_BLEND));
      GL_CHECK(glDisable(GL_CULL_FACE));

      displayBackgroundShader.bind();
      compute->copyTexture(cameraInterface->getCurrentFrame(), &displayBackgroundBuffer.getComputeTexture());
      displayBackgroundVertex.bind();
      displayBackgroundShader.activateTexture("backgroundTexture", 0, displayBackgroundBuffer.getGraphicsTexture());
      GL_CHECK(glDrawArrays(GL_TRIANGLES, 0, 6));
      displayBackgroundVertex.unbind();
      displayBackgroundShader.unbind();
    }
  }

  GL_CHECK(glEnable(GL_DEPTH_TEST));
  GL_CHECK(glDepthFunc(GL_LESS));
  GL_CHECK(glFrontFace(GL_CW));
  GL_CHECK(glCullFace(GL_BACK));

  for (uint solver = 0; solver < SOLVER_MAX; solver++)
  {
    if (solversUint[solver])
    {
      const uint elements = solversUint[solver]->lastPartition().end();

      if (!elements) continue;

      // copy particle position and collision data for display
      float* collisionData = (float*)&((*solversUint[solver]->particleCollisionData.host())[0]);
      ParticleStruct* particles = &(*(solversUint[solver]->particles.host()))[0];

      if (getFrameOption(RENDER_PARTICLES_OPTION).boolValue)
      {
        GL_CHECK(glEnable(GL_CULL_FACE));
        // copy particle position and collision data for display
        displayPositionBuffer.copyData((float*)particles, elements * sizeof(ParticleStruct));
        displayCollisionBuffer.copyData((float*)collisionData, elements * sizeof(ParticleCollisionData));

        displayParticleShader.bind();
        displayParticleVertex.bind();
        displayParticleElements.bind();
        GL_CHECK(glDrawElementsInstanced(GL_TRIANGLES, displayParticleElements.count(), GL_UNSIGNED_INT, NULL, elements));
        displayParticleElements.unbind();
        displayParticleVertex.unbind();
        displayParticleShader.unbind();

        //TODO: Debug why line is not working with cloth rendering on when z vector is 0
        displayLineShader.bind();
        displayLineVertex.bind();
        GL_CHECK(glDrawArraysInstanced(GL_LINES, 0, 2, elements));
        displayLineVertex.unbind();
        displayLineShader.unbind();
        GL_CHECK(glDisable(GL_CULL_FACE));
      }

      if (getFrameOption(RENDER_SOLIDS_OPTION).boolValue && solver != SOLVER_FLUID)
      {
        displaySolidShader.bind();
        displaySolidVertex.bind();

        for (const PartitionInfo &partition : *(solversUint[solver]->partitions.host()))
        {
          GL_CHECK(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 4, particles + partition.offset));
          GL_CHECK(glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 4, collisionData + 4 * partition.offset));
          IdentityInfo identity = solversUint[solver]->particles.host()->at(partition.offset).identity;
          PhysicsEntity* entity = entities[solver][getEntityId(identity)];

          displaySolidShader.set("fillShader", 1.f);
          if (entity->displayElements.count())
          {
            entity->displayElements.bind();
            GL_CHECK(glDrawElementsInstanced(GL_TRIANGLES, entity->displayElements.count(), GL_UNSIGNED_INT, NULL, 1));
            entity->displayElements.unbind();
          }

          displaySolidShader.set("fillShader", 0.f);
          if (entity->displayEdges.count())
          {
            entity->displayEdges.bind();
            GL_CHECK(glDrawElementsInstanced(GL_LINES, entity->displayEdges.count(), GL_UNSIGNED_INT, NULL, 1));
            entity->displayEdges.unbind();
          }
        }
        displaySolidVertex.unbind();
        displaySolidShader.unbind();
      }

      if (getFrameOption(RENDER_SOLIDS_OPTION).boolValue && solver == SOLVER_FLUID)
      {
        displayFlatShader.bind();

        // copy particle position and collision data for display
        displayPositionBuffer.copyData((float*)particles, elements * sizeof(ParticleStruct));
        displayCollisionBuffer.copyData((float*)collisionData, elements * sizeof(ParticleCollisionData));

        displayFlatShader.set("fillShader", 1.f);

        GL_CHECK(glEnable(GL_BLEND));
        displayFlatVertex.bind();
        for (const PartitionInfo &partition : *(solversUint[solver]->partitions.host()))
        {
          GL_CHECK(glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, displayFlatVertex.count(), partition.count));
        }
        displayFlatVertex.unbind();
        GL_CHECK(glDisable(GL_BLEND));
        displayFlatShader.unbind();
      }
    }
  }

  GL_CHECK(glEnable(GL_BLEND));
  GL_CHECK(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

  displayBoxShader.bind();
  displayBoxVertex.bind();
  displayBoxElements.bind();
  // render boundign boxes if supplied by the colision solver
  if (getFrameOption(RENDER_BOUNDING_BOXES_OPTION).boolValue && collisionSolver->particleGroupBoundingBoxes.size())
  {
    DeviceArray<XAB>* collisionBoundingBoxes = &collisionSolver->particleGroupBoundingBoxes;
    displayBoxBuffer.copyData((float*)&((*collisionBoundingBoxes->host())[0]), (uint)collisionBoundingBoxes->host()->size() * sizeof(XAB));

    displayBoxBuffer.bind();
    GL_CHECK(glEnableVertexAttribArray(1));
    GL_CHECK(glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 8, 0));
    GL_CHECK(glVertexAttribDivisor(1, 1));
    GL_CHECK(glEnableVertexAttribArray(2));
    GL_CHECK(glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 8, (const GLvoid*)(sizeof(GLfloat) * 4)));
    GL_CHECK(glVertexAttribDivisor(2, 1));
    GL_CHECK(glDrawElementsInstanced(GL_LINES, displayBoxElements.count(), GL_UNSIGNED_INT, NULL, (uint)collisionBoundingBoxes->host()->size()));
  }

  // render boundign boxes if supplied by the colision solver
  if (getFrameOption(RENDER_SYSTEM_BOUND_OPTION).boolValue && collisionSolver->systemBoundingBox.size())
  {
    DeviceArray<XAB>* collisionBoundingBoxes = &collisionSolver->systemBoundingBox;
    displayBoxBuffer.copyData((float*)&((*collisionBoundingBoxes->host())[0]), (uint)collisionBoundingBoxes->host()->size() * sizeof(XAB));

    displayBoxBuffer.bind();
    GL_CHECK(glEnableVertexAttribArray(1));
    GL_CHECK(glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 8, 0));
    GL_CHECK(glVertexAttribDivisor(1, 1));
    GL_CHECK(glEnableVertexAttribArray(2));
    GL_CHECK(glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 8, (const GLvoid*)(sizeof(GLfloat) * 4)));
    GL_CHECK(glVertexAttribDivisor(2, 1));
    GL_CHECK(glDrawElementsInstanced(GL_LINES, displayBoxElements.count(), GL_UNSIGNED_INT, NULL, (uint)collisionBoundingBoxes->host()->size()));
  }
  displayBoxElements.unbind();
  displayBoxVertex.unbind();
  displayBoxShader.unbind();

  if (getFrameOption(RENDER_GRID_HEATMAP_OPTION).boolValue && ((UniformGridCollisionSolver*)collisionSolver)->gridCellParticleCount.size())
  {
    displayGridShader.bind();
    displayBoxVertex.bind();
    displayGridElements.bind();

    DeviceArray <uint>* gridCellParticleCount = &((UniformGridCollisionSolver*)collisionSolver)->gridCellParticleCount;

    displayGridBuffer.copy((float*)&((*gridCellParticleCount->host())[0]), 0, 0, (uint)gridCellParticleCount->host()->size() / 4);
    displayGridShader.activateTexture("gridCellParticleCount", 0, displayGridBuffer);
    Real3 systemMin = collisionSolver->systemBoundingBox.host()->at(0).min;
    displayGridShader.set("systemMin", systemMin.x, systemMin.y, systemMin.z, 0.f);
    displayGridShader.set("maxRadius", 1.f/((UniformGridCollisionSolver*)collisionSolver)->invMaxRadius.host()->at(0));
    displayGridShader.set("gridSize", (int)((UniformGridCollisionSolver*)collisionSolver)->gridSize);
    displayGridShader.set("totalParticles", (float)instanceNodeCount);

    GL_CHECK(glDrawElementsInstanced(GL_TRIANGLES, displayGridElements.count(), GL_UNSIGNED_INT, NULL, (uint)gridCellParticleCount->host()->size()));

    displayGridElements.unbind();
    displayBoxVertex.unbind();
    displayGridShader.unbind();
  }

  GL_CHECK(glDisable(GL_BLEND));
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
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_SHARED)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_AUX)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTITIONS)->get(),
    allocator->getHeap(COMPUTE_HEAP_SECTIONS)->get(),
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
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_SHARED)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTICLE_AUX)->get(),
    allocator->getHeap(COMPUTE_HEAP_PARTITIONS)->get(),
    allocator->getHeap(COMPUTE_HEAP_SECTIONS)->get(),
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

    float zero = 0.f;
    ComputeUtil::get(0)->clearBuffer(compute, allocator->getHeap(COMPUTE_HEAP_PARTICLE_DIFF)->get(), instanceNodeCount * sizeof(ParticleStruct)/sizeof(uint), *((uint*)&zero));

    updates.clear();

#ifdef ENABLE_RENDERING
    const uint textureWidth = 128;

    displayParticleVertex.gen();
    displaySolidVertex.gen();
    displayBoxVertex.gen();
    displayFlatVertex.gen();
    displayLineVertex.gen();
    displayBackgroundVertex.gen();

    displayParticleElements.gen();
    displayBoxElements.gen();
    displayGridElements.gen();

    displayPositionBuffer.gen();
    displayCollisionBuffer.gen();
    displayBoxBuffer.gen();

    {
      const uint gridElements = (((UniformGridCollisionSolver*)collisionSolver)->gridSize * mSqr(((UniformGridCollisionSolver*)collisionSolver)->gridSize)) / 4;
      const uint textureHeight = (gridElements + textureWidth - 1) / textureWidth;

      displayGridBuffer.init(textureWidth, textureHeight);
      displayGridBuffer.gen();
    }

    clearColor[0] = 0.7f;
    clearColor[1] = 0.7f;
    clearColor[2] = 0.7f;
    clearColor[3] = 1.0f;

    displayParticleShader.init("ParticleVert.glsl", "ParticleFrag.glsl");
    displaySolidShader.init("SolidVert.glsl", "SolidFrag.glsl");
    displayFlatShader.init("FlatVert.glsl", "SolidFrag.glsl");
    displayBoxShader.init("BoxVert.glsl", "BoxFrag.glsl");
    displayLineShader.init("LineVert.glsl", "LineFrag.glsl");
    displayGridShader.init("GridVert.glsl", "GridFrag.glsl");
    displayBackgroundShader.init("BGVert.glsl", "BGFrag.glsl");

    displayParticleShader.linkPrograms();
    displaySolidShader.linkPrograms();
    displayFlatShader.linkPrograms();
    displayBoxShader.linkPrograms();
    displayLineShader.linkPrograms();
    displayGridShader.linkPrograms();
    displayBackgroundShader.linkPrograms();
    displayBackgroundShader.bind();
    displayBackgroundShader.set("flipY", (int)1);
    displayBackgroundShader.set("fillScreen", (int)0);
    displayBackgroundShader.unbind();

    createSphere(1.f);

    displaySolidVertex.bind();
    GL_CHECK(glEnableVertexAttribArray(0));
    displaySolidVertex.unbind();

    createUnitCircle();

    createUnitBox();

    float line[] = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };
    displayLineVertex.copyData(line, 2, 0, 3 * sizeof(float));
    displayLineVertex.bind();
    displayPositionBuffer.bind();
    GL_CHECK(glEnableVertexAttribArray(0));
    GL_CHECK(glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0));
    GL_CHECK(glVertexAttribDivisor(0, 1));
    displayCollisionBuffer.bind();
    GL_CHECK(glEnableVertexAttribArray(1));
    GL_CHECK(glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0));
    GL_CHECK(glVertexAttribDivisor(1, 1));
    displayLineVertex.unbind();

    float quad[] = {
      -1.f, -1.f, 0.f, 1.f, 0.f, 0.f,
      -1.f,  1.f, 0.f, 1.f, 0.f, 1.f,
       1.f, -1.f, 0.f, 1.f, 1.f, 0.f,
       1.f, -1.f, 0.f, 1.f, 1.f, 0.f,
      -1.f,  1.f, 0.f, 1.f, 0.f, 1.f,
       1.f,  1.f, 0.f, 1.f, 1.f, 1.f,
    };
    displayBackgroundVertex.copyData(quad, 6, 0, 6 * sizeof(float));
    displayBackgroundVertex.bind();
    GL_CHECK(glEnableVertexAttribArray(0));
    GL_CHECK(glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 6 * sizeof(float), 0));
    GL_CHECK(glEnableVertexAttribArray(1));
    GL_CHECK(glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 6 * sizeof(float), 0));
    displayBackgroundVertex.unbind();
#endif

    indexMap.resize(instanceNodeCount, false);
  }

  // update gravity if
  if (down.length() > 0.f)
  {
    down.normalize();
    down *= 9.8f;
    setGravity(down);
  }

  // skip the below steps if simulation paused
  if (getFrameOption(PAUSE_SIM_OPTION).boolValue)
  {
    return;
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

  collisionSolver->solve(instanceNodeCount, systemSettings.device());

  for (uint i = 0; i < SOLVER_MAX; i++)
  {
    if (solversUint[i])
    {
      solversUint[i]->solve(timeStep);
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

void PhysicsSystem::setCameraInterface(CameraInterface *cameraInterface)
{
  this->cameraInterface = cameraInterface;
}
