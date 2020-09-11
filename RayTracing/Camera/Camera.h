#ifndef CAMERA_H
#define CAMERA_H

#include <Common/RayTracingStruct.h>
#include <Entities/RayTracingEntity.h>

/*!
@class Base class for camera, implementing pinhole camera.
*/
class CameraSimple : public CameraStruct, public RayTracingEntity
{
  friend class ReaderScene;

protected:

  float scale;
  uint  samples;        // samples per pixel
  real  nearPlane;

  Byte* buffer;

  real  sampleIntensity;

  // Function to calculate differentials for different axis and setup ray origin
  void calculateDelta(Real3& origin);

  void update();

public:

  CameraSimple(ComputeInterface* compute);

  ~CameraSimple();

  /*!@function Update camera struct values using these window based values.*/
  virtual void update(const Real3& origin, const Real3& cameraUp, const Real3& cameraFront);

  virtual void emitPrimaryRays(DeviceArray<Ray>& rays);
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

  ~CameraDepth();

  virtual void update(const Real3& origin, const Real3& cameraUp, const Real3& cameraFront);

  virtual void emitPrimaryRays(DeviceArray<Ray>& rays);
};

#endif
