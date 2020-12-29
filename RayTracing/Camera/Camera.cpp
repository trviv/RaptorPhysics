#include "Camera.h"

//#define DEBUG_RT_CAMERA
#define RT_CAMERA_BUFFER_SIZE 3

Camera::Camera(ComputeInterface* compute)
  :RayTracingEntity(compute)
{
  scale = 1.f;
  samples = 1;
  buffer = NULL;

  includeFiles.push_back("ComputeHeader.shader");
  includeFiles.push_back("ComputeShared.h");
  includeFiles.push_back("RayStructs.h");
  includeFiles.push_back("RayTracingStruct.h");

  vector<string> oldType = { "RayStruct"};
  vector<string> newType = { getRayStructName(RayStructPositionDirectionColor) };
  registerShader(compute, "Camera.shader", &oldType, &newType);
  kernels.push_back(programs[0].createKernel("emitPrimaryRaysZWalkLocal"));

  deviceData = new DeviceArray<uint>[RT_CAMERA_BUFFER_SIZE];
  for (uint i=0; i<RT_CAMERA_BUFFER_SIZE; i++)
  {
    deviceData[i].create(compute);
    deviceData[i].resize(sizeof(CameraStruct) / sizeof(uint), false);
    deviceData[i].host()->resize(sizeof(CameraStruct) / sizeof(uint), false);
  }

  rayCount.create(compute);
  rayCount.resize(4, false);

  frameIndex = 0;
  deviceIndex = 0;
}

Camera::~Camera()
{
  if(buffer)
  {
    delete buffer;
  }
  for (uint i=0; deviceData && i<RT_CAMERA_BUFFER_SIZE; i++)
  {
    deviceData[i].free();
  }
  if (deviceData)
  {
    delete deviceData;
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
  logComputeError("Camera object does not carry an identity!");
  return IdentityInfo_t();
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

  // compare current matrix to the previous matrix and set frame index to 0 if its changed
  CameraStruct* prevValue = (CameraStruct*)&(*(deviceData[deviceIndex].host()))[0];
  for (uint i=0; i<16; i++)
  {
    if (this->viewMatrixInv[i] != prevValue->viewMatrixInv[i])
    {
      frameIndex = 0;
      break;
    }
  }

  deviceIndex = (deviceIndex + 1) % RT_CAMERA_BUFFER_SIZE;
  memcpy((CameraStruct*)&(*(deviceData[deviceIndex].host()))[0], (CameraStruct*)this, sizeof(CameraStruct));
  deviceData[deviceIndex].syncDevice();
  frameIndex++;
}

void Camera::emitPrimaryRays(DeviceArray<uint>& rays, RayStructType rayType)
{
  rays.resize(width * height * getRayStructSize(rayType) / 4, false);

  // create primary rays
  {
    size_t workgroupSize[3], workgroupCount[3];
    uint size[3] = {width, height, 1};
    compute->configureSize(workgroupSize, workgroupCount, size, 1024);

    ComputeMemory* buffers[] = {
      rays.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    kernels[0].setArgs(buffers, bufferCount);
    kernels[0].setArg(deviceData->device(), bufferCount);

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

const ComputeMemory* Camera::getDeviceCamera()const
{
  return deviceData[deviceIndex].device();
}
