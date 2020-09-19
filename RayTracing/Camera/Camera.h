#ifndef CAMERA_H
#define CAMERA_H

#include <Common/RayTracingStruct.h>
#include <Entities/RayTracingEntity.h>

/*!
@class Base class for camera, implementing pinhole camera.
*/
class Camera : public CameraStruct, public RayTracingEntity
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

  Camera(ComputeInterface* compute);

  ~Camera();

  /*!@function Update camera struct values using these matrices.*/
  virtual void update(const real projectionMatrix[], const real modelviewMatrix[]);

  virtual void emitPrimaryRays(DeviceArray<uint>& rays, RayStructType rayType);

  virtual void setScale(real scale);
};


/*!
@class Class implementing lens camera.
*/
class CameraDepth : public Camera
{
protected:

  real focus;     // focus of lens
  real aperture;  // camera aperture

public:

  CameraDepth(ComputeInterface* compute);

  ~CameraDepth();

  virtual void update(const real projectionMatrix[16], const real modelviewMatrix[16]);

  virtual void emitPrimaryRays(DeviceArray<uint>& rays, RayStructType rayType);
};

#endif
