#ifndef SCENE_H
#define SCENE_H

#include <Core.h>
#include <Light/Light.h>
#include <Camera/Camera.h>

/*!
@class Class representing a ray tracing system.
*/
class RayTracingSystem : protected ShaderEntity
{
  friend class ReaderScene;

protected:

  ComputeInterface*     compute;
  RayTracingAllocator*  allocator;

  /*!@member Camera used in the scene.*/
  CameraSimple*             camera;

  /*!@member Lights in the scene.*/
  DeviceArray<LightStruct>  lights;

  /*!@member Ray buffer for the scene.*/
  DeviceArray<Ray>          rays;

public:

  RayTracingSystem();

  ~RayTracingSystem();

  void init(ComputeInterface* compute);

  void update();

  void renderParticles(Window* window, ComputeMemory* particles, uint count);
};

#endif
