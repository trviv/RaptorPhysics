#include "MainSystem.h"

static Clock physicsSystemClock;

#define FRAME_BUFFERING_SIZE 3
#define GUI_REFRESH_AFTER_FRAMES 0xF

MainSystem::MainSystem(ComputeInterface* compute)
  :compute(compute), cameraInterface(compute)
{
}

MainSystem::~MainSystem()
{
}

void MainSystem::init(int argc, char** argv, int width, int height, const char* name)
{
  Window::init(argc, argv, width, height, name);
  frameCount = 0;
  frameCaptureStart = 0;
  frameCaptureEnd = 0;

#ifdef ENABLE_RENDERING
  initRender();
#endif
}

void MainSystem::createFromFile(const char fileName[])
{
  reader.readFile(this, fileName);
}

#ifdef ENABLE_RENDERING

void MainSystem::initRender()
{
  displayGridBuffer = Texture(TEXTURE_FORMAT_INT);
  rayTracingOutBuffer = Texture(TEXTURE_FORMAT_HALF);

  optionFrame->addElement(new UIElement(RENDER_PARTICLES_OPTION, true, "fa-solid-900", 0xF141));
  optionFrame->addElement(new UIElement(RENDER_SOLIDS_OPTION, true, "fa-solid-900", 0xF1B3));
  optionFrame->addElement(new UIElement(RENDER_BOUNDING_BOXES_OPTION, true, "fa-brands-400", 0xF247));
  optionFrame->addElement(new UIElement(RENDER_SYSTEM_BOUND_OPTION, true, "fa-brands-400", 0xF1CB));
  optionFrame->addElement(new UIElement(RENDER_GRID_HEATMAP_OPTION, true, "fa-solid-900", 0xF37F));
  optionFrame->addElement(new UIElement(RENDER_RESET_CAMERA_OPTION, RENDER_RESET_CAMERA_OPTION, "fa-solid-900", 0xF03D));

  timeSliderFrame->addElement(new UIElement(REPLAY_SIM_OPTION, 0, 0, "fa-solid-900"));

  elapsedRenderTime = 0.f;

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
  displayDensityBuffer.gen();

  clearColor[0] = 0.7f;
  clearColor[1] = 0.7f;
  clearColor[2] = 0.7f;
  clearColor[3] = 1.0f;

  displayParticleShader.init("ParticleVert.glsl", "ParticleFrag.glsl");
  displaySolidShader.init("SolidVert.glsl", "SolidFrag.glsl");
  displayFlatShader.init("FlatVert.glsl", "SolidFrag.glsl");
  displayBoxShader.init("BoxVert.glsl", "PassthruFrag.glsl");
  displayLineShader.init("LineVert.glsl", "PassthruFrag.glsl");
  displayGridShader.init("GridVert.glsl", "PassthruFrag.glsl");
  displayBackgroundShader.init("BGVert.glsl", "BGFrag.glsl");
  displayRayTraceShader.init("BGVert.glsl", "BGFrag.glsl");

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
  displayFlatShader.bind();
  displayFlatShader.set("screenAligned", (int)1);
  displayFlatShader.set("fillShader", 1.f);
  displayFlatShader.unbind();

  displayRayTraceShader.linkPrograms();
  displayRayTraceShader.bind();
  displayRayTraceShader.set("flipY", (int)1);
  displayRayTraceShader.set("fillScreen", (int)0);
  displayRayTraceShader.unbind();

  createSphere(1.f);

  displaySolidVertex.bind();
  GL_CHECK(glEnableVertexAttribArray(0));
  GL_CHECK(glEnableVertexAttribArray(1));
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
  GL_CHECK(glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(4 * sizeof(float))));
  displayBackgroundVertex.unbind();
}

void MainSystem::createSphere(float radius)
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

void MainSystem::createUnitBox()
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

void MainSystem::createUnitCircle()
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
  displayDensityBuffer.bind();
  GL_CHECK(glEnableVertexAttribArray(3));
  GL_CHECK(glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(float), 0));
  GL_CHECK(glVertexAttribDivisor(3, 1));
  displayFlatVertex.unbind();
}

void MainSystem::render()
{
  // initialize ray tracer if it has no primitives
  if (rayTracingSystem.getPrimCount() == 0)
  {
    for (uint s = 0; s < SOLVER_MAX; s++)
    {
      const auto solver = physicsSystem.getSolver((SolverType)s);
      if (solver)
      {
        uint elements = solver->lastPartition().end();

        if (!elements) continue;

        vector<pair<uint, pair<uint, uint>>> entityCounts;

        uint prevEntityId = -1;

        // loop through partitions and add unique entities with offset and count
        for (const auto& partition : *solver->getPartitions().host())
        {
          const IdentityInfo entityIdentity = solver->getParticles().host()->at(partition.offset).identity;
          uint entityId = entityIdentity.identity^getInstanceId(entityIdentity);
          if (prevEntityId == entityId)
          {
            entityCounts.back().second.second += partition.count;
          }
          else
          {
            entityCounts.push_back(pair<uint, pair<uint, uint>>(entityId, pair<uint, uint>(partition.offset, partition.count)));
          }
        }

        for (const auto& entityCount : entityCounts)
        {
          PrimitiveArrayEntity *entity = new PrimitiveArrayEntity(RayTracingEntitySpheres, entityCount.second.second);
          entity->setAttribute(EntityPrimitiveAttributePosition, solver->getParticles().device(), PackingInfo(entityCount.second.first));
          entity->setAttribute(EntityPrimitiveAttributeRadius, solver->getParticleCollisionData().device(), PackingInfo(entityCount.second.first, 4, 3));
          entity->setMaterialId(entityMaterialMap[entityCount.first]);
          rayTracingSystem.registerAndInstantiateEntity(entity);
        }
      }
    }

    {
      bottomSurface.create(compute);

      XAB systemBound = physicsSystem.getSystemSettings().systemBound;
//      systemBound.min.x = -1000.f;
//      systemBound.min.z = -1000.f;
//
//      systemBound.max.x = 1000.f;
//      systemBound.max.z = 1000.f;

      PrimitiveStruct pos;
      pos.position = systemBound.min;     bottomSurface.host()->push_back(pos);
      pos.position.z = systemBound.max.z; bottomSurface.host()->push_back(pos);
      pos.position.x = systemBound.max.x; bottomSurface.host()->push_back(pos);

      bottomSurface.host()->push_back(pos);
      pos.position.z = systemBound.min.z; bottomSurface.host()->push_back(pos);
      pos.position.x = systemBound.min.x; bottomSurface.host()->push_back(pos);

      bottomSurface.syncDevice();
      PrimitiveArrayEntity *entity = new PrimitiveArrayEntity(RayTracingEntityTriangles, 2);
      entity->setAttribute(EntityPrimitiveAttributePosition, bottomSurface.device(), PackingInfo());
      entity->setMaterialId((*entityMaterialMap.begin()).second);
      rayTracingSystem.registerAndInstantiateEntity(entity);
    }

    rayTracingSystem.commit();
  }

  if (true)
  {
    rayTracingSystem.updateCamera(this->projectionMatrix, this->modelMatrix);
    rayTracingSystem.render();

    const uint camWidth  = rayTracingSystem.getCameraStruct().width;
    const uint camHeight = rayTracingSystem.getCameraStruct().height;

    // initialize ray tracing output buffer
    if (rayTracingOutBuffer.get() == -1)
    {
      rayTracingOutBuffer.init(camWidth , camHeight);
      rayTracingOutBuffer.gen();

      displayRayTraceShader.bind();
      displayRayTraceShader.set("frameDimensions", (float)width(), (float)height(),
                                (float)camWidth,(float)camHeight);
      displayRayTraceShader.unbind();
    }

    GL_CHECK(glDisable(GL_DEPTH_TEST));
    GL_CHECK(glDisable(GL_BLEND));
    GL_CHECK(glDisable(GL_CULL_FACE));

    const uint hostOffset = ((frameCount + FRAME_BUFFERING_SIZE - 1) % FRAME_BUFFERING_SIZE) * camWidth * camHeight;
    rayTracingSystem.getColorOutputBuffer().syncHost(hostOffset, camWidth * camHeight);

    const colorType4* colorOutputBuffer = &(*(rayTracingSystem.getColorOutputBuffer().host()))[hostOffset];

    displayRayTraceShader.bind();
    rayTracingOutBuffer.copy((float*)colorOutputBuffer);
    displayBackgroundVertex.bind();
    displayRayTraceShader.activateTexture("backgroundTexture", 0, rayTracingOutBuffer);
    GL_CHECK(glDrawArrays(GL_TRIANGLES, 0, 6));
    displayBackgroundVertex.unbind();
    displayRayTraceShader.unbind();
  }

  // sync all output buffers
  for (uint s = 0; s < SOLVER_MAX; s++)
  {
    const auto solver = physicsSystem.getSolver((SolverType)s);
    if (solver)
    {
      uint elements = solver->lastPartition().end();

      if (!elements) continue;

      const uint hostOffset = ((frameCount + FRAME_BUFFERING_SIZE - 1) % FRAME_BUFFERING_SIZE) * elements;
      solver->getParticles().syncHost(hostOffset, elements);
      solver->getParticleCollisionData().syncHost(hostOffset, elements);

      if (s == SOLVER_FLUID)
      {
        ((FluidSolver*)solver)->getParticlesDensity().syncHost(hostOffset, elements);
      }
    }
  }

  const CollisionSolver* collisionSolver = physicsSystem.getCollisionSolver();
  uint instanceNodeCount = physicsSystem.particleCount();

  if (optionFrame->getElement(RENDER_BOUNDING_BOXES_OPTION)->boolValue && collisionSolver->particleGroupBoundingBoxes.size())
  {
    collisionSolver->particleGroupBoundingBoxes.syncHost();
  }

  if (optionFrame->getElement(RENDER_SYSTEM_BOUND_OPTION)->boolValue && collisionSolver->systemBoundingBox.size())
  {
    collisionSolver->systemBoundingBox.syncHost();
  }

  if (optionFrame->getElement(RENDER_GRID_HEATMAP_OPTION)->boolValue && ((UniformGridCollisionSolver*)collisionSolver)->gridCellParticleCount.size())
  {
    ((UniformGridCollisionSolver*)collisionSolver)->gridCellParticleCount.syncHost();
    collisionSolver->systemBoundingBox.syncHost();
  }

  compute->sync(false);

  // Display frame info
  if ((frameCount & GUI_REFRESH_AFTER_FRAMES) == 0 || forceRefreshUICount)
  {
    char temp[64];
    sprintf(temp, "Particles:   %d\n", instanceNodeCount);
    statFrame->setText(temp);

    uint vertexCount = 0;
    if (optionFrame->getElement(RENDER_PARTICLES_OPTION)->boolValue)
    {
      vertexCount += displayParticleVertex.count() * instanceNodeCount;
      vertexCount += displayLineVertex.count() * instanceNodeCount;
    }
    if (optionFrame->getElement(RENDER_SOLIDS_OPTION)->boolValue)
    {
      // TODO: Find a good way to find this value, ignore for now
      //vertexCount += displaySolidVertex.count();
    }
    if (optionFrame->getElement(RENDER_BOUNDING_BOXES_OPTION)->boolValue && collisionSolver->particleGroupBoundingBoxes.size())
    {
      vertexCount += displayBoxVertex.count() * collisionSolver->particleGroupBoundingBoxes.host()->size();
    }
    if (optionFrame->getElement(RENDER_SYSTEM_BOUND_OPTION)->boolValue && collisionSolver->systemBoundingBox.size())
    {
      vertexCount += displayBoxVertex.count() * collisionSolver->systemBoundingBox.host()->size();
    }
    if (optionFrame->getElement(RENDER_GRID_HEATMAP_OPTION)->boolValue && ((UniformGridCollisionSolver*)collisionSolver)->gridCellParticleCount.size())
    {
      vertexCount += displayBoxVertex.count() * ((UniformGridCollisionSolver*)collisionSolver)->gridCellParticleCount.size();
    }
    sprintf(temp, "Vertices:    %d\n", vertexCount);
    statFrame->setText(statFrame->getText() + temp);
    sprintf(temp, "Sim Time:    %.1f ms\n", elapsedSimTime / (((frameCount-1) & GUI_REFRESH_AFTER_FRAMES) + 1));
    statFrame->setText(statFrame->getText() + temp);
    sprintf(temp, "Render Time: %.1f ms\n", elapsedRenderTime / (((frameCount-1) & GUI_REFRESH_AFTER_FRAMES) + 1));
    statFrame->setText(statFrame->getText() + temp);

    // only clear if triggered by refresh cycle and not force UI count
    if ((frameCount & GUI_REFRESH_AFTER_FRAMES) == 0)
    {
      elapsedSimTime = 0.f;
      elapsedRenderTime = 0.f;
    }
    forceRefreshUICount = max(forceRefreshUICount - 1, 0);
  }
  if ((frameCount & GUI_REFRESH_AFTER_FRAMES) == GUI_REFRESH_AFTER_FRAMES)
  {
    memoryManager.dealloc();
  }

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

  if (cameraInterface.isActive())
  {
    if (displayBackgroundBuffer.getComputeTexture() == NULL)
    {
      uint cameraSize[2] = {cameraInterface.width(), cameraInterface.height()};
      displayBackgroundBuffer = createSharedTexture(compute, cameraSize, SHARED_TEXTURE_FORMAT_UINT8x4);
      displayBackgroundShader.bind();
      displayBackgroundShader.set("frameDimensions", (float)width(), (float)height(),
                                  (float)displayBackgroundBuffer.getGraphicsTexture().width(),
                                  (float)displayBackgroundBuffer.getGraphicsTexture().height());
      displayBackgroundShader.unbind();
    }

    if (cameraInterface.getCurrentFrame())
    {
      GL_CHECK(glDisable(GL_DEPTH_TEST));
      GL_CHECK(glDisable(GL_BLEND));
      GL_CHECK(glDisable(GL_CULL_FACE));

      displayBackgroundShader.bind();
      compute->copyTexture(cameraInterface.getCurrentFrame(), &displayBackgroundBuffer.getComputeTexture());
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

  // only when running and not paused
  if (timeSliderFrame->isShrunk())
  {
    // add new entry to position stream
    particlePositionStream.addItem(instanceNodeCount * sizeof(ParticleStruct));
    ((UIElement*)timeSliderFrame->getElement(REPLAY_SIM_OPTION))->rangeValue = 0;
  }
  else
  {
    // set time scroll range based on recorder output when paused
    UIElement* timeline = (UIElement*)timeSliderFrame->getElement(REPLAY_SIM_OPTION);
    timeline->range[0] = -(particlePositionStream.size()-1);
    timeline->range[1] = 0;
  }

  // just an easy way to keep track of particle offsets per solver
  uint instanceStartingOffset = 0;

  for (uint s = 0; s < SOLVER_MAX; s++)
  {
    const auto solver = physicsSystem.getSolver((SolverType)s);
    if (solver)
    {
      const uint elements = solver->lastPartition().end();

      if (!elements) continue;

      // copy particle position and collision data for display
      const uint hostOffset = (frameCount % FRAME_BUFFERING_SIZE) * elements;
      const ParticleCollisionData* collisionData = &(*(solver->getParticleCollisionData().host()))[hostOffset];
      const ParticleStruct* particles = &(*(solver->getParticles().host()))[hostOffset];

      // only when running and not paused
      if (timeSliderFrame->isShrunk())
      {
        // append data at the end of position stream
        particlePositionStream.appendToLast(particles, elements * sizeof(ParticleStruct));
      }
      else
      {
        uint offset = ((UIElement*)timeSliderFrame->getElement(REPLAY_SIM_OPTION))->rangeValue;
        particles = (ParticleStruct*)particlePositionStream.getItem(particlePositionStream.size() + offset - 1, instanceStartingOffset * sizeof(ParticleStruct));
      }

      if (optionFrame->getElement(RENDER_PARTICLES_OPTION)->boolValue)
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

      if (optionFrame->getElement(RENDER_SOLIDS_OPTION)->boolValue && s != SOLVER_FLUID)
      {
        displaySolidShader.bind();
        displaySolidVertex.bind();

        displayPositionBuffer.copyData((float*)particles, elements * sizeof(ParticleStruct));
        displayCollisionBuffer.copyData((float*)collisionData, elements * sizeof(ParticleCollisionData));

        for (const PartitionInfo &partition : *(solver->getPartitions().host()))
        {
          displayPositionBuffer.bind();
          GL_CHECK(glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(ParticleStruct), (void*)(sizeof(ParticleStruct) * partition.offset)));
          displayCollisionBuffer.bind();
          GL_CHECK(glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(ParticleCollisionData), (void*)(sizeof(ParticleCollisionData) * partition.offset)));

          IdentityInfo identity = solver->getParticles().host()->at(partition.offset).identity;
          const PhysicsEntity* entity = physicsSystem.getEntities((SolverType)s)[getEntityId(identity)];

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

      if (optionFrame->getElement(RENDER_SOLIDS_OPTION)->boolValue && s == SOLVER_FLUID)
      {
        displayFlatShader.bind();
        const float* density = &(*(((FluidSolver*)solver)->getParticlesDensity().host()))[hostOffset];

        // copy particle position and collision data for display
        displayPositionBuffer.copyData((float*)particles, elements * sizeof(ParticleStruct));
        displayCollisionBuffer.copyData((float*)collisionData, elements * sizeof(ParticleCollisionData));
        displayDensityBuffer.copyData(density, elements * sizeof(float));

        GL_CHECK(glEnable(GL_BLEND));
        displayFlatVertex.bind();
        float invRestDensity = ((FluidSolver*)solver)->getEntitySharedData().host()->at(0).invRestDensity;
        displayFlatShader.set("invRestDensity", invRestDensity);
        for (const PartitionInfo &partition : *(solver->getPartitions().host()))
        {
          GL_CHECK(glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, displayFlatVertex.count(), partition.count));
        }
        displayFlatVertex.unbind();
        GL_CHECK(glDisable(GL_BLEND));
        displayFlatShader.unbind();
      }

      instanceStartingOffset += elements;
    }
  }

  GL_CHECK(glEnable(GL_BLEND));
  GL_CHECK(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

  displayBoxShader.bind();
  displayBoxVertex.bind();
  displayBoxElements.bind();
  // render boundign boxes if supplied by the colision solver
  if (optionFrame->getElement(RENDER_BOUNDING_BOXES_OPTION)->boolValue && collisionSolver->particleGroupBoundingBoxes.size())
  {
    const DeviceArray<XAB>* collisionBoundingBoxes = &collisionSolver->particleGroupBoundingBoxes;
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
  if (optionFrame->getElement(RENDER_SYSTEM_BOUND_OPTION)->boolValue && collisionSolver->systemBoundingBox.size())
  {
    const DeviceArray<XAB>* collisionBoundingBoxes = &collisionSolver->systemBoundingBox;
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

  if (optionFrame->getElement(RENDER_GRID_HEATMAP_OPTION)->boolValue && ((UniformGridCollisionSolver*)collisionSolver)->gridCellParticleCount.size())
  {
    displayGridShader.bind();
    displayBoxVertex.bind();
    displayGridElements.bind();

    DeviceArray <uint>* gridCellParticleCount = &((UniformGridCollisionSolver*)collisionSolver)->gridCellParticleCount;

    const uint gridElements = (((UniformGridCollisionSolver*)collisionSolver)->gridSize * mSqr(((UniformGridCollisionSolver*)collisionSolver)->gridSize)) / 4;
    if ((displayGridBuffer.height() * displayGridBuffer.width()) < gridElements)
    {
      const uint textureWidth = 128;
      const uint textureHeight = (gridElements + textureWidth - 1) / textureWidth;
      displayGridBuffer.init(textureWidth, textureHeight);
      displayGridBuffer.gen();
    }

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

  // don't increase frame count if paused
  if (timeSliderFrame->isShrunk())
  {
    frameCount++;
  }
}
#endif

void MainSystem::step()
{
  const float renderTime = physicsSystemClock.getTimeMilliseconds();
  physicsSystemClock.reset();

  // update gravity if
  if (down.length() > 0.f)
  {
    down.normalize();
    down *= Real3(physicsSystem.getSystemSettings().gravity).length();

    // disable orientation with frame capture
    if (!frameCaptureStart)
      physicsSystem.setGravity(down);
  }

  // skip the below steps if simulation paused
  if (!timeSliderFrame->isShrunk())
  {
    return;
  }

  physicsSystem.step();

  if (frameCaptureStart)
  {
    if (frameCount == frameCaptureStart)
    {
      compute->sync(true);
      compute->startCapture();
    }
    else if (frameCount == frameCaptureEnd)
    {
      compute->sync(true);
      compute->endCapture();
    }
  }

  elapsedSimTime += physicsSystemClock.getTimeMilliseconds();
#ifdef ENABLE_RENDERING
  elapsedRenderTime += renderTime;
#endif

  physicsSystemClock.reset();
}
