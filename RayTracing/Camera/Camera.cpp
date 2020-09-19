#include "Camera.h"

Camera::Camera(ComputeInterface* compute)
  :RayTracingEntity(RayTracingEntityCamera, compute)
{
  scale = 1.f;
  samples = 1;
  buffer = NULL;

  includeFiles.push_back("ComputeHeader.shader");
  includeFiles.push_back("ComputeShared.h");
  includeFiles.push_back("RayStructs.h");
  includeFiles.push_back("RayTracingStruct.h");

  registerShader(compute, "Camera.shader", NULL, NULL);
  kernels.push_back(programs[0].createKernel("emitPrimaryRays"));
}

Camera::~Camera()
{
  if(buffer)
  {
    delete buffer;
  }
}

void Camera::calculateDelta(Real3& origin)
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

void Camera::update()
{
  logComputeError("Camera update without arguments is not supported!");
}

void Camera::update(const real projectionMatrix[16], const real modelviewMatrix[16])
{
  const uint initWidth  = this->width;
  const uint initHeight = this->height;
  const real w = width, h = height;

  real invProjection[16], invModelview[16];
  Matrix4::invert(invProjection, projectionMatrix);
  Matrix4::invert(invModelview, modelviewMatrix);

  const real frontVec[] = {0.f, 0.f, -1.f, 0.f};
  Real3 cameraFront = Matrix4::transformVec(modelviewMatrix, frontVec);

  const real upVec[] = {0.f, 1.f, 0.f, 0.f};
  Real3 cameraUp = Matrix4::transformVec(modelviewMatrix, upVec);

  Real3 cross = cameraFront.cross(cameraUp);
  cross.normalize();

  const real posVec[] = {0.f, 0.f, 0.f, 1.f};
  this->origin = Matrix4::transformVec(invModelview, posVec);

  real topLeftVec[] = {-1.f, 1.f, 0.f, 1.f};
  Matrix4::transformVec(topLeftVec, invProjection, topLeftVec);
  topLeftVec[3] = 1.f;
  this->topLeft = Matrix4::transformVec(invModelview, topLeftVec);

  this->deltaX  = cross / w;
  this->deltaY  = (Real3(0) - cameraUp) / h;
  this->width   = initWidth;
  this->height  = initHeight;
}

void Camera::emitPrimaryRays(DeviceArray<uint>& rays, RayStructType rayType)
{
  rays.resize(width * height * getRayStructSize(rayType) / 4, false);

  // create primary rays
  {
    size_t workgroupSize[3], workgroupCount[3];
    uint size[3] = {width, height, 1};
    compute->configureSize(workgroupSize, workgroupCount, size);

    ComputeMemory* buffers[] = {
      rays.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[0].setArgs(buffers, bufferCount);
    kernels[0].setArg<CameraStruct>(this, bufferCount);

    compute->execute(kernels[0], workgroupSize, workgroupCount);

    rays.syncHost();
    compute->sync(true);
  }
}

void Camera::setScale(real scale)
{
  width  = width * scale;
  height = height * scale;
}
