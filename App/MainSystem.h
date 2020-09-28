#ifndef RENDERER_H
#define RENDERER_H

#include <Core.h>
#include <UnifiedPhysics.h>
#include <RayTracing.h>
#include "ReaderScene.h"

static const string REPLAY_SIM_OPTION             = "Replay Sim";
static const string RENDER_PARTICLES_OPTION       = "Particles";
static const string RENDER_SOLIDS_OPTION          = "Solids";
static const string RENDER_BOUNDING_BOXES_OPTION  = "Bounding Boxes";
static const string RENDER_SYSTEM_BOUND_OPTION    = "Scene Box";
static const string RENDER_GRID_HEATMAP_OPTION    = "Grid Heatmap";
static const string RENDER_RESET_CAMERA_OPTION    = "Reset Camera";

/*!
@class Class to render contents of the scene.
*/
class MainSystem : public Window
{
protected:

  uint  frameCount;
  uint  frameCaptureStart;
  uint  frameCaptureEnd;
  real  elapsedSimTime;

#ifdef ENABLE_RENDERING

  Buffer  displayPositionBuffer;
  Buffer  displayCollisionBuffer;
  Buffer  displayBoxBuffer;
  Buffer  displayDensityBuffer;
  Texture displayGridBuffer;
  ComputeGraphicsSharedTexture displayBackgroundBuffer;
  Texture rayTracingOutBuffer;

  Vertex  displayParticleVertex;
  Vertex  displaySolidVertex;
  Vertex  displayFlatVertex;
  Vertex  displayBoxVertex;
  Vertex  displayLineVertex;
  Vertex  displayBackgroundVertex;

  Shader  displayParticleShader;
  Shader  displaySolidShader;
  Shader  displayFlatShader;
  Shader  displayBoxShader;
  Shader  displayGridShader;
  Shader  displayLineShader;
  Shader  displayBackgroundShader;
  Shader  displayRayTraceShader;

  Face    displayParticleElements;
  Face    displayBoxElements;
  Face    displayGridElements;

  real    elapsedRenderTime;

  void createSphere(float radius);

  void createUnitBox();

  void createUnitCircle();

  void initRender();

  void render();

#endif

  ComputeInterface* compute;
  ReaderScene       reader;

  /*!@member Camera Interface.*/
  CameraInterface   cameraInterface;
  PhysicsSystem     physicsSystem;

  /*!@member Memory streamer for particle positions.*/
  MemoryStreamer    particlePositionStream;

  RayTracingSystem  rayTracingSystem;

  DeviceArray<PrimitiveStruct> bottomSurface;

  friend class ReaderScene;

public:

  MainSystem(ComputeInterface* compute);

  ~MainSystem();

  void init(int argc, char** argv, int width = 512, int height = 512, const char* name = "GL Window");

  void createFromFile(const char fileName[]);

  void step();
};

#endif

