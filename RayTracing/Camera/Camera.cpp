#include "Camera.h"

//#define DEBUG_RT_CAMERA

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

void Camera::update()
{
  logComputeError("Camera update without arguments is not supported!");
}

void Camera::update(const real projectionMatrix[16], const real modelviewMatrix[16])
{
  this->scale = tan(60.f * 0.5f * M_PI / 180.f);
  Matrix4::invert(this->viewMatrixInv, modelviewMatrix);
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

#ifdef DEBUG_RT_CAMERA
    rays.syncHost();
    compute->sync(true);
    vector<Ray>& rt = (vector<Ray>&)(*rays.host());
#endif
  }
}

void Camera::setScale(real scale)
{
  width  = width * scale;
  height = height * scale;
}
