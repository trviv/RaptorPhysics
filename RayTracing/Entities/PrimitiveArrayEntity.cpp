#include "RayTracingEntity.h"
#include <sstream>

PrimitiveArrayEntity::PrimitiveArrayEntity(RayTracingEntityType type, uint primitiveCount, ComputeInterface* compute)
  :RayTracingEntity(compute)
{
  this->identity.identity = 0;
  setRayTracingEntityId(this->identity, type, 0);
  this->primInfo.primType = type;
  this->primitiveCount    = primitiveCount;
}

PrimitiveArrayEntity::~PrimitiveArrayEntity()
{
  
}

RayTracingEntity* PrimitiveArrayEntity::createCopy()const
{
  PrimitiveArrayEntity *newEntity = new PrimitiveArrayEntity((RayTracingEntityType)this->primInfo.primType, primitiveCount);
  *newEntity = *this;
  return newEntity;
}

void PrimitiveArrayEntity::createBox(const real dim[])
{
  if (!deviceData)
    deviceData = new DeviceArray<uint>(compute);

  const bool rectangle = (dim[2] == 0.f);
  const real boxVertices[] = {
    -dim[0], -dim[1], -dim[2], 0.f, +dim[0], -dim[1], -dim[2], 0.f, -dim[0], +dim[1], -dim[2], 0.f, +dim[0], +dim[1], -dim[2], 0.f,
    -dim[0], -dim[1], +dim[2], 0.f, +dim[0], -dim[1], +dim[2], 0.f, -dim[0], +dim[1], +dim[2], 0.f, +dim[0], +dim[1], +dim[2], 0.f,
  };
  const uint indices[] = {0, 1, 2, 2, 1, 3, 4, 1, 0, 5, 1, 4, 0, 2, 4, 4, 2, 6, 6, 5, 4, 7, 5, 6, 2, 3, 6, 6, 3, 7, 5, 3, 1, 7, 3, 5};

  deviceData->host()->clear();
  for (int i=0; i<(rectangle?16:32); i++)
  {
    deviceData->host()->push_back(((uint*)boxVertices)[i]);
  }
  for (int i=0; i<(rectangle?6:36); i++)
  {
    deviceData->host()->push_back(indices[i]);
  }
  deviceData->syncDevice();

  setAttribute(EntityPrimitiveAttributePosition, deviceData->device(), PackingInfo());
  setAttribute(EntityPrimitiveAttributeIndex, deviceData->device(), PackingInfo(rectangle?16:32, 1));
  primitiveCount = (rectangle?2:12);
}

void PrimitiveArrayEntity::createSphere(const real radius)
{
  if (!deviceData)
    deviceData = new DeviceArray<uint>(compute);

  const real sphereCenter[] = {0.f, 0.f, 0.f, 0.f};
  const real sphereRadius[] = {radius};

  deviceData->host()->clear();
  for (auto i : sphereCenter)
  {
    deviceData->host()->push_back(*((uint*)&i));
  }
  for (auto i : sphereRadius)
  {
    deviceData->host()->push_back(*((uint*)&i));
  }
  deviceData->syncDevice();

  setAttribute(EntityPrimitiveAttributePosition, deviceData->device(), PackingInfo());
  setAttribute(EntityPrimitiveAttributeRadius, deviceData->device(), PackingInfo(sizeof(sphereCenter)/sizeof(real), 1));
  primitiveCount = 1;
}

void PrimitiveArrayEntity::createMesh(const string fileName)
{
  if (!deviceData)
    deviceData = new DeviceArray<uint>(compute);

  uint triangles = 0;
  uint vertices = 0;

  deviceData->host()->clear();

  std::stringstream ss;
  std::string data = IOInterface::readFile(fileName.c_str());
  ss << data;

  std::string temp;

  while (std::getline(ss, temp))
  {
    std::stringstream line;
    line << temp;

    char head;
    line >> head;
    switch (head)
    {
      case '#':
        continue;

      case 'v':
      {
        for (int i=0; i<3; i++)
        {
          float val;
          line >> val;
          deviceData->host()->push_back(*((uint*)&val));
        }
        deviceData->host()->push_back(0);
        vertices++;
      }
        break;
      case 'f':
      {
        for (int i=0; i<3; i++)
        {
          uint val;
          line >> val;
          deviceData->host()->push_back(val-1);
        }

        // change triangle orientation
        {
          uint temp = deviceData->host()->at(deviceData->host()->size()-2);
          deviceData->host()->at(deviceData->host()->size()-2) = deviceData->host()->at(deviceData->host()->size()-1);
          deviceData->host()->at(deviceData->host()->size()-1) = temp;
        }
        triangles++;
      }
        break;
      default:
        break;
    }
  }
  deviceData->syncDevice();

  setAttribute(EntityPrimitiveAttributePosition, deviceData->device(), PackingInfo());
  setAttribute(EntityPrimitiveAttributeIndex, deviceData->device(), PackingInfo(vertices*4, 1));
  primitiveCount = triangles;
}

RayTracingEntityId PrimitiveArrayEntity::getIdentity()const
{
  return identity;
}

uint PrimitiveArrayEntity::getPrimCount()const
{
  return primitiveCount;
}

void PrimitiveArrayEntity::update()
{

}
