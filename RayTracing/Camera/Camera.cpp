#include "Camera.h"

CameraSimple::CameraSimple(ComputeInterface* compute)
  :RayTracingEntity(RayTracingEntityCamera, compute)
{
  samples = 1;
  buffer = NULL;

  includeFiles.push_back("ComputeHeader.shader");
  includeFiles.push_back("ComputeShared.h");
  includeFiles.push_back("RayTracingStruct.h");

  vector<string> newType = { "uint", "float", "float3" };
  vector<string> oldType = { "IndexType", "CoefficientType", "VariableType" };
  registerShader(compute, "Camera.shader", &oldType, &newType);
  kernels.push_back(programs[0].createKernel("emitPrimaryRays"));
}

CameraSimple::~CameraSimple()
{
  if(buffer)
  {
    delete buffer;
  }
}

void CameraSimple::calculateDelta(Real3& origin)
{
  const real w = width, h = height;
  sampleIntensity = 1.f / mSqr(samples);

  const Matrix4& trans = affine[TRANS];

  // calculation of screen coordinates
  // the one with lower value will be of unit length
  if (h < w)
  {
    topLeft.x = -w / h; topLeft.y = 1;
  }
  else
  {
    topLeft.x = -1; topLeft.y = h / w;
  }

  // set camera vertices in local coordinates
  deltaX.x = -topLeft.x;
  deltaX.y = topLeft.y;
  deltaY.x = topLeft.x;
  deltaY.y = -topLeft.y;
  topLeft.z = deltaX.z = deltaY.z = -nearPlane;

  // transform to world coordinates
  trans.transformPos(topLeft);
  trans.transformPos(deltaX);
  trans.transformPos(deltaY);
  trans.transformPos(origin);

  // calculate the shifts per pixel
  deltaX = (Real3(deltaX) - topLeft) / w;
  deltaY = (Real3(deltaY) - topLeft) / h;
}

void CameraSimple::update()
{
  Real3(origin).setNull();
  // get origin and set differentials
  calculateDelta(origin);
}

Ray CameraSimple::sampleCamera(const Real3& screenCoordinates)const
{
  Ray worldRay;
  worldRay.origin = origin;
  worldRay.direction = Real3(topLeft) + Real3(deltaX) * width * screenCoordinates[0] + Real3(deltaY) * height * screenCoordinates[1] - origin;
  Real3(worldRay.direction).normalize();
  return worldRay;
}

void CameraSimple::emitPrimaryRays(DeviceArray<Ray>* rays)
{
  
}
