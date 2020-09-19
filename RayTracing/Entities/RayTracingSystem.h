#ifndef SCENE_H
#define SCENE_H

#include <Core.h>
#include <Light/Light.h>
#include <Camera/Camera.h>
#include <Acceleration/AccelerationDataStruct.h>

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
  Camera* camera;

  /*!@member Lights in the scene.*/
  DeviceArray<LightStruct>  lights;

  /*!@member Ray buffer for the scene.*/
  DeviceArray<uint>         rays;

  /*!@member Ray hit information buffer for the scene.*/
  DeviceArray<uint>         hits;

  /*!@member Acceleration struct for the system.*/
  AccelerationDataStruct*   accelerationStruct;

public:

  RayTracingSystem();

  ~RayTracingSystem();

  /*!@function Initialize the system based on max number of rays.*/
  void init(ComputeInterface* compute, const uint maxRays);

  /*!@function Get primitives in ray tracing system.*/
  uint getPrimCount()const;

  /*!
  @function Register sphere buffer to the system.
  @param primitiveBuffer Should be a device array similar to or of type PositionStruct_t.
  */
  void registerSphereBuffer(const ComputeMemory* primitiveBuffer, const ComputeMemory* radiusBuffer, PackingInfo radiusInfo, uint count);

  /*!@function Update camera based on given matrices.*/
  void updateCamera(const real projectionMatrix[16], const real modelviewMatrix[16]);

  void render();
};

#endif
