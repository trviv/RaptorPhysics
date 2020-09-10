#ifndef SCENE_H
#define SCENE_H

#include <Light/Light.h>
#include <Camera/Camera.h>

/*!
@class Class representng a scene.
*/
class Scene : protected ShaderEntity
{
  friend class ReaderScene;

protected:

  ComputeInterface*     compute;
  RayTracingAllocator*  allocator;

  /*!@member Camera used in the scene.*/
  CameraSimple*             camera;

  /*!@member Lights in the scene.*/
  DeviceArray<LightStruct>  lights;

public:

  Scene();

  ~Scene();

  void init(ComputeInterface* compute);

  void update();
};

#endif
