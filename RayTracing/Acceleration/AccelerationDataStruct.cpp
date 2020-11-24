#include "AccelerationDataStruct.h"

string getIntersectionTypeName(IntersectionType type)
{
  switch (type)
  {
    case IntersectionTypeClosest:
      return "IntersectionTypeClosest";
      break;
    case IntersectionTypeAny:
      return "IntersectionTypeAny";
      break;
    default:
      return "";
      break;
  }

  return "";
}

uint AccelerationDataStruct::accXABComputeUtilId = -1;
uint AccelerationDataStruct::sortComputeUtilId   = -1;

//#define DEBUG_ACCELERATION_DATA_STRUCT

AccelerationDataStruct::AccelerationDataStruct()
{
}

AccelerationDataStruct::~AccelerationDataStruct()
{
}

void AccelerationDataStruct::create(ComputeInterface* compute)
{
  this->compute = compute;
  includeFiles.push_back("ComputeHeader.shader");
  includeFiles.push_back("ComputeShared.h");
  includeFiles.push_back("RayTracingStruct.h");

  registerShader(compute, "AccelerationDataStructCreate.shader", NULL, NULL);

  createPrimitiveBoundingBoxes = programs[0].createKernel("createPrimitiveBoundingBoxes");

  includeFiles.push_back("RayStructs.h");
  includeFiles.push_back("HitStructs.h");

  for (int i=0; i<IntersectionTypeMax; i++)
  {
    for (int r=0; r<RayStructTypeMax; r++)
    {
      for (int h=0; h<HitStructTypeMax; h++)
      {
        vector<string> oldType = {getIntersectionTypeName((IntersectionType)i), "RayStruct", "HitStruct"};
        vector<string> newType = {"", getRayStructName((RayStructType)r), getHitStructName((HitStructType)h)};
        getRayStructDefines(oldType, newType, (RayStructType)r);
        getHitStructDefines(oldType, newType, (HitStructType)h);
        registerShader(compute, "AccelerationDataStructTraverse.shader", &oldType, &newType);
        intersectRayKernels[i][r][h] = programs.back().createKernel("intersectRays");
      }
    }
  }

  boundingBoxes.create(compute);

  accXABComputeUtilId = ComputeUtil::getXABUtil(compute);
  sortComputeUtilId   = ComputeUtil::getUIntUtil(compute);

  primitiveCount = 0;
  vertexCount    = 0;
}

uint AccelerationDataStruct::getPrimCount()const
{
  return primitiveCount;
}

void AccelerationDataStruct::commit(const DeviceArray<PrimitiveStruct>* vertexArray, const DeviceArray<PrimitiveAttrib>* attributeArray,
                                    DeviceArray<RTSystemSettings>* systemSettings)
{
  this->vertexArray     = vertexArray;
  this->attributeArray  = attributeArray;
  this->systemSettings  = systemSettings;

  primitiveCount = decodePrimitiveInfo(systemSettings->host()->at(0).globalOffsets[RTPrimitiveCount-1]).indexOffset;
  vertexCount    = decodePrimitiveInfo(systemSettings->host()->at(0).globalOffsets[RTPrimitiveCount-1]).vertexOffset;

  boundingBoxes.resize(primitiveCount, false);
}

void AccelerationDataStruct::fullBuild()
{
  uint primBatchSize = 8;

  {
    uint primBatchCount = mAlignBy(primitiveCount, primBatchSize);

    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, primBatchCount);

    createPrimitiveBoundingBoxes.setArg(boundingBoxes.device(),   0);
    createPrimitiveBoundingBoxes.setArg(vertexArray->device(),    1);
    createPrimitiveBoundingBoxes.setArg(attributeArray->device(), 2);
    createPrimitiveBoundingBoxes.setArg(systemSettings->device(), 3);
    createPrimitiveBoundingBoxes.setArg(&primBatchSize,           4);
    createPrimitiveBoundingBoxes.setArg(&primitiveCount,          5);

    compute->execute(createPrimitiveBoundingBoxes, workgroupSize, workgroupCount);

#ifdef DEBUG_ACCELERATION_DATA_STRUCT
    boundingBoxes.syncHost();
    compute->sync();
#endif
  }
}

void AccelerationDataStruct::intersectRays(ComputeMemory* hits, HitStructType hitType, ComputeMemory* rays, RayStructType rayType,
                                           uint rayCount, IntersectionType intersectionType)
{
  {
    ComputeKernel& intersectionKernel = intersectRayKernels[intersectionType][rayType][hitType];

    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, rayCount);

    intersectionKernel.setArg(hits, 0);
    intersectionKernel.setArg(rays, 1);
    intersectionKernel.setArg(&rayCount, 2);
    intersectionKernel.setArg(boundingBoxes.device(),   3);
    intersectionKernel.setArg(vertexArray->device(),    4);
    intersectionKernel.setArg(attributeArray->device(), 5);
    intersectionKernel.setArg(&primitiveCount,          6);
    intersectionKernel.setArg(systemSettings->device(), 7);

    compute->execute(intersectionKernel, workgroupSize, workgroupCount);

#ifdef DEBUG_ACCELERATION_DATA_STRUCT
    boundingBoxes.syncHost();
    compute->sync();
#endif
  }
}
