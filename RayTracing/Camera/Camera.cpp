#include "Camera.h"

CameraSimple::CameraSimple(ComputeInterface* compute)
  :RayTracingEntity(RayTracingEntityCamera, compute)
{
  scale = 1.f;
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
  logComputeError("Camera update without arguments is not supported!");
}

void CameraSimple::update(const Real3& origin, const Real3& cameraUp, const Real3& cameraFront)
{
  const real w = width, h = height;

  Real3 cross = cameraFront.cross(cameraUp);
  cross.normalize();

  if (h < w)
  {
    this->topLeft = cross - (w / h) * cameraUp;
  }
  else
  {
    this->topLeft = cross * (h / w) - cameraUp;
  }

  this->origin = origin;
  this->deltaX = cameraUp;
  this->deltaY = cross;
}

void CameraSimple::emitPrimaryRays(DeviceArray<Ray>& rays)
{
  rays.resize(width * height, false);

  { // get count for each grid cell
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
  }
}
