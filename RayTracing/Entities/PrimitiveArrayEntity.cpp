#include "RayTracingEntity.h"
#include <sstream>

PrimitiveArrayEntity::PrimitiveArrayEntity(RayTracingEntityType type, uint primitiveCount, ComputeInterface* compute)
  :RayTracingEntity(compute)
{
  this->identity.identity = 0;
  setRayTracingEntityId(this->identity, type, 0);
  this->primInfo.primType = type;
  this->primInfo.primitiveCount = primitiveCount;
  this->primBound.min = Real3(-1.f, -1.f, -1.f);
  this->primBound.max = Real3(1.f, 1.f, 1.f);
}

PrimitiveArrayEntity::~PrimitiveArrayEntity()
{
  
}

RayTracingEntity* PrimitiveArrayEntity::createCopy()const
{
  PrimitiveArrayEntity *newEntity = new PrimitiveArrayEntity((RayTracingEntityType)this->primInfo.primType, primInfo.primitiveCount);
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
  setAttribute(EntityPrimitiveAttributeIndex, deviceData->device(), PackingInfo(rectangle?16:32, 3));
  primInfo.primitiveCount = (rectangle?2:12);
  primInfo.indexCount = (rectangle?4:8);
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
  primInfo.primitiveCount = 1;
  primInfo.indexCount = 1;
}

struct triangleIndices
{
  int i1, i2, i3;
};

bool compTriangleIndices(triangleIndices &a, triangleIndices &b)
{
  if (a.i1 < b.i1) return true;
  if (a.i1 > b.i1) return false;
  if (a.i2 < b.i2) return true;
  if (a.i2 > b.i2) return false;
  if (a.i3 < b.i3) return true;
  if (a.i3 > b.i3) return false;
  return false;
}

void PrimitiveArrayEntity::createMesh(const string fileName)
{
  if (!deviceData)
    deviceData = new DeviceArray<uint>(compute);

  uint triangles = 0;
  uint vertices = 0;

  vector<triangleIndices> allindices;

  deviceData->host()->clear();

  std::stringstream ss;
  std::string data = IOInterface::readFile(fileName.c_str());
  ss << data;

  std::string temp;

  Real3 vertexMin(FLT_MAX);
  Real3 vertexMax(-FLT_MAX);

  while (std::getline(ss, temp))
  {
    std::stringstream line;
    line << temp;

    string head;
    line >> head;

    if (head == "v")
    {
      Real3 vert;
      for (int i=0; i<3; i++)
      {
        float val;
        line >> val;
        deviceData->host()->push_back(*((uint*)&val));
        vert[i] = val;
      }
      vertexMin = vertexMin.min(vert);
      vertexMax = vertexMax.max(vert);
      deviceData->host()->push_back(0);
      vertices++;
    }
    else if (head == "f")
    {
      int indices[6] = {0};
      int i = 0;
      while (line)
      {
        string indexStr;
        line >> indexStr;

        if (indexStr == "")
          break;

        uint val = atoi(indexStr.substr(0, indexStr.find("/")).c_str());

        indices[i++] = val-1;
      }
      triangles++;

      // add another triangle for quad prim
      if (i == 4)
      {
        indices[4] = indices[2];
        indices[5] = indices[0];
        triangles++;
        i = 6;
      }

      uint temp = indices[1];
      indices[1] = indices[2];
      indices[2] = temp;

      if (i >= 3)
      {
        allindices.push_back(triangleIndices{indices[0],indices[1],indices[2]});
      }
      if (i == 6)
      {
        allindices.push_back(triangleIndices{indices[3],indices[4],indices[5]});
      }
    }
  }

  sort(allindices.begin(), allindices.end(), compTriangleIndices);

  for (int j=0; j<allindices.size(); j++)
  {
    deviceData->host()->push_back(allindices[j].i1);
    deviceData->host()->push_back(allindices[j].i2);
    deviceData->host()->push_back(allindices[j].i3);
  }

  primBound.min = vertexMin;
  primBound.max = vertexMax;
  deviceData->syncDevice();

  setAttribute(EntityPrimitiveAttributePosition, deviceData->device(), PackingInfo());
  setAttribute(EntityPrimitiveAttributeIndex, deviceData->device(), PackingInfo(vertices*4, 3));
  primInfo.primitiveCount = triangles;
  primInfo.indexCount = vertices;
}

RayTracingEntityId PrimitiveArrayEntity::getIdentity()const
{
  return identity;
}

const XAB& PrimitiveArrayEntity::getPrimBound()const
{
  return primBound;
}

void PrimitiveArrayEntity::update()
{

}
