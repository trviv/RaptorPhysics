#ifndef CAMERA_H
#define CAMERA_H

#include <Common/RayTracingStruct.h>
#include <Entity/RayTracingEntity.h>

/*!
@class Base class for camera, implementing pinhole camera.
*/
class CameraSimple : public CameraStruct, public RayTracingEntity
{
  friend class ReaderScene;

protected:

  uint  width, height;
  uint  samples;        // samples per pixel
  real  nearPlane;

  Byte* buffer;

  real  sampleIntensity;

  // Function to calculate differentials for different axis and setup ray origin
  void calculateDelta(Real3& origin);

public:

  CameraSimple(ComputeInterface* compute);

  ~CameraSimple();

  void update();

  /*!@function Sample a ray at this coordinate.*/
  Ray sampleCamera(const Real3& screenCoordinates)const;

  virtual void emitPrimaryRays(DeviceArray<Ray>* rays);
};


/*!
@class Class implementing lens camera.
*/
class CameraDepth : public CameraSimple
{

protected:

  real focus;     // focus of lens
  real aperture;  // camera aperture

public:

  CameraDepth(ComputeInterface* compute);

  void update();
};

#endif
