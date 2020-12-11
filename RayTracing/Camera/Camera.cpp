#include "Camera.h"

//#define DEBUG_RT_CAMERA

Camera::Camera(ComputeInterface* compute)
  :RayTracingEntity(compute)
{
  setRayTracingEntityId(this->identity, RayTracingEntityCamera, 0);
  scale = 1.f;
  samples = 1;
  buffer = NULL;
  setRayTracingEntityId(identity, RayTracingEntityCamera, 0);

  includeFiles.push_back("ComputeHeader.shader");
  includeFiles.push_back("ComputeShared.h");
  includeFiles.push_back("RayStructs.h");
  includeFiles.push_back("RayTracingStruct.h");

  vector<string> oldType = { "RayStruct"};
  vector<string> newType = { getRayStructName(RayStructPositionDirectionColor) };
  registerShader(compute, "Camera.shader", &oldType, &newType);
  kernels.push_back(programs[0].createKernel("emitPrimaryRays"));

  deviceData.create(compute);
  deviceData.resize(1, false);

  rayCount.create(compute);
  rayCount.resize(4, false);
}

Camera::~Camera()
{
  if(buffer)
  {
    delete buffer;
  }
}

RayTracingEntity* Camera::createCopy()const
{
  Camera *newCamera = new Camera(this->compute);
  *newCamera = *this;
  return newCamera;
}

RayTracingEntityId Camera::getIdentity()const
{
  return identity;
}

const DeviceArray<uint>* Camera::getRayCount()const
{
  return &rayCount;
}

void Camera::update()
{
  logComputeError("Camera update without arguments is not supported!");
}

void Camera::update(const real projectionMatrix[16], const real modelviewMatrix[16])
{
  this->scale = tan(60.f * 0.5f * M_PI / 180.f);
  Matrix4::invert(this->viewMatrixInv, modelviewMatrix);
  compute->copyFromHost(deviceData.device(), 0, sizeof(CameraStruct), (CameraStruct*)this, true);
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
    kernels[0].setArg(deviceData.device(), bufferCount);

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

  rayCount.host()->clear();
  rayCount.host()->reserve(4);
  rayCount.host()->push_back(width*height);
  rayCount.host()->push_back(1);
  rayCount.host()->push_back(1);
  rayCount.host()->push_back(0);
  rayCount.syncDevice();
}
