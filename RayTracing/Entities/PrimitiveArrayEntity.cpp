#include "RayTracingEntity.h"
#include <sstream>

PrimitiveArrayEntity::PrimitiveArrayEntity(RayTracingEntityType type, uint primitiveCount, ComputeInterface* compute)
  :RayTracingEntity(compute)
{
  this->identity.identity = 0;
  setRayTracingEntityId(this->identity, type, 0);
  this->primInfo.primitiveType  = type;
  this->primInfo.primitiveCount = primitiveCount;
  this->primBound.min = Real3(-1.f, -1.f, -1.f);
  this->primBound.max = Real3(1.f, 1.f, 1.f);
}

PrimitiveArrayEntity::~PrimitiveArrayEntity()
{
  
}

RayTracingEntity* PrimitiveArrayEntity::createCopy()const
{
  PrimitiveArrayEntity *newEntity = new PrimitiveArrayEntity((RayTracingEntityType)this->primInfo.primitiveType, primInfo.primitiveCount);
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

  uint vertexOffset = 0;
  uint vertexAndIndexOffset = 0;

  deviceData->host()->clear();
  for (int i=0; i<(rectangle?16:32); i++)
  {
    deviceData->host()->push_back(((uint*)boxVertices)[i]);
  }
  vertexOffset = (rectangle?4:8) * sizeof(PrimitiveStruct) / 4;

  for (int i=0; i<(rectangle?6:36); i++)
  {
    deviceData->host()->push_back(indices[i]);
    if ((i%3) == 2)
    {
      deviceData->host()->push_back(-1);
    }
  }
  vertexAndIndexOffset = vertexOffset + (rectangle?8:48);

  primInfo.primitiveCount = (rectangle?2:12);
  primInfo.vertexCount    = (rectangle?4:8);

  generateNeighbourBasedNormal(rectangle?4:8);
  deviceData->syncDevice();

  setAttribute(EntityPrimitiveAttributePosition,  deviceData->device(), PackingInfo());
  setAttribute(EntityPrimitiveAttributeIndex,     deviceData->device(), PackingInfo(vertexOffset, 4));
  setAttribute(EntityPrimitiveAttributeNormal,    deviceData->device(), PackingInfo(vertexAndIndexOffset, 4));
}

void PrimitiveArrayEntity::createSphere(const real radius)
{
  if (!deviceData)
    deviceData = new DeviceArray<uint>(compute);

  const real sphereCenter[] = {0.f, 0.f, 0.f, 0.f};
  const real sphereRadius[] = {0.f, 0.f, 0.f, radius};

  uint vertexOffset = 0;
  uint vertexAndIndexOffset = 0;

  deviceData->host()->clear();
  for (auto i : sphereCenter)
  {
    deviceData->host()->push_back(*((uint*)&i));
  }
  vertexOffset = 1 * sizeof(PrimitiveStruct) / 4;

  for (auto i : sphereRadius)
  {
    deviceData->host()->push_back(*((uint*)&i));
  }
  vertexAndIndexOffset = vertexOffset + 4;

  deviceData->syncDevice();

  setAttribute(EntityPrimitiveAttributePosition,  deviceData->device(), PackingInfo());
  setAttribute(EntityPrimitiveAttributeRadius,    deviceData->device(), PackingInfo(vertexOffset, 4));
  setAttribute(EntityPrimitiveAttributeNormal,    deviceData->device(), PackingInfo(vertexAndIndexOffset, 4));

  primInfo.primitiveCount = 1;
  primInfo.vertexCount    = 1;
}

struct DEFAULT_ALIGN PrimitiveIndices
{
  uint i0, i1, i2, i3;
};

bool compTriangleIndices(PrimitiveIndices &a, PrimitiveIndices &b)
{
  if (a.i0 < b.i0) return true;
  if (a.i0 > b.i0) return false;
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

  vector<PrimitiveIndices> primitiveIndices;
  vector<Real3> normals;

  uint vertexOffset = 0;
  uint vertexAndIndexOffset = 0;

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
    else if (head == "vn")
    {
      Real3 normal;
      for (int i=0; i<3; i++)
      {
        float val;
        line >> val;
        normal[i] = val;
      }
      normals.push_back(normal);
    }
    else if (head == "f")
    {
      uint indices[4];
      indices[3] = 0xFFFFFFFF;
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
        primInfo.primitiveType = RayTracingEntityIndexedQuads;
      }
      primitiveIndices.push_back(PrimitiveIndices{indices[0], indices[1], indices[2], indices[3]});
    }
  }

  sort(primitiveIndices.begin(), primitiveIndices.end(), compTriangleIndices);

  for (int j=0; j<primitiveIndices.size(); j++)
  {
    deviceData->host()->push_back(primitiveIndices[j].i0);
    deviceData->host()->push_back(primitiveIndices[j].i1);
    deviceData->host()->push_back(primitiveIndices[j].i2);
    deviceData->host()->push_back(primitiveIndices[j].i3);
  }

  primBound.min = vertexMin;
  primBound.max = vertexMax;

  primInfo.primitiveCount = triangles;
  primInfo.vertexCount    = vertices;

  vertexOffset = vertices * sizeof(PrimitiveStruct) / 4;
  vertexAndIndexOffset = vertexOffset + (uint)primitiveIndices.size() * sizeof(PrimitiveIndices) / 4;

  generateNeighbourBasedNormal(vertices, &normals);
  deviceData->syncDevice();

  setAttribute(EntityPrimitiveAttributePosition, deviceData->device(), PackingInfo());
  setAttribute(EntityPrimitiveAttributeIndex, deviceData->device(), PackingInfo(vertexOffset, 4));
  setAttribute(EntityPrimitiveAttributeNormal, deviceData->device(), PackingInfo(vertexAndIndexOffset, 4));
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

void PrimitiveArrayEntity::generateNeighbourBasedNormal(uint vertexCount, vector<Real3>* normals)
{
  std::vector<Real3> smoothNormals;

  if (normals == NULL || normals->size() == 0)
  {
    const auto& hostBuffer = *deviceData->host();
    const auto* hostVertexBuffer = (PrimitiveStruct*)&hostBuffer[0];

    PrimitiveStruct*  vertex  = (PrimitiveStruct*)&hostVertexBuffer[0];
    PrimitiveIndices* indices = (PrimitiveIndices*)(&hostVertexBuffer[vertexCount]);

    smoothNormals = std::vector<Real3>(vertexCount, Real3(0.f));
    normals       = &smoothNormals;

    for (uint i=0; i<primInfo.primitiveCount; i++)
    {
      const PrimitiveIndices index = indices[i];
      assert(index.i0 < vertexCount && index.i0 < vertexCount && index.i2 < vertexCount && (index.i3 == -1 || index.i3 < vertexCount));

      Real3 vert0 = vertex[index.i0].position;
      Real3 edge1 = Real3(vertex[index.i1].position) - vert0;
      Real3 edge2 = Real3(vertex[index.i2].position) - vert0;

      Real3 normal = edge1.cross(edge2);
      normal.normalize();

      smoothNormals[index.i0] += normal;
      smoothNormals[index.i1] += normal;
      smoothNormals[index.i2] += normal;

      if (index.i3 == -1) continue;

      vert0 = vertex[index.i0].position;
      edge1 = Real3(vertex[index.i2].position) - vert0;
      edge2 = Real3(vertex[index.i3].position) - vert0;

      normal = edge1.cross(edge2);
      normal.normalize();

      smoothNormals[index.i0] += normal;
      smoothNormals[index.i2] += normal;
      smoothNormals[index.i3] += normal;
    }
  }

  for (int i=0; i<vertexCount; i++)
  {
    Real3 normal = normals->at(i);
    float length = normal.length();
    if (length > 0)
    {
      normal /= length;

    }
    deviceData->host()->push_back(*((uint*)&normal.x));
    deviceData->host()->push_back(*((uint*)&normal.y));
    deviceData->host()->push_back(*((uint*)&normal.z));
    deviceData->host()->push_back(0);
  }
}
