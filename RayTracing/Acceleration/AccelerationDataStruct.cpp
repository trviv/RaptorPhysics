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
  compute               = NULL;
  pointerVertexArray    = NULL;
  pointerAttributeArray = NULL;
  pointerSystemSettings = NULL;
  needsRebuild          = true;
}

AccelerationDataStruct::~AccelerationDataStruct()
{
}

void AccelerationDataStruct::validateBuild()const
{
  if (needsRebuild)
  {
    logComputeError("Acceleration Data Structure need to be build, before using!");
  }
}

void AccelerationDataStruct::updatePointers()
{
  pointerLeafNodeBoundingBoxes = leafNodeBoundingBoxes.device();
}

void AccelerationDataStruct::registerCreateShaders(const vector<string>* oldType, const vector<string>* newType)
{
  includeFiles.push_back("ComputeHeader.shader");
  includeFiles.push_back("ComputeShared.h");
  includeFiles.push_back("RayTracingStruct.h");

  registerShader(compute, "AccelerationDataStructCreate.shader", oldType, newType);

  createPrimitiveBoundingBoxes = programs.back().createKernel("createPrimitiveBoundingBoxes");
}

void AccelerationDataStruct::registerTraverseShaders(const vector<string>* oldTypeArg, const vector<string>* newTypeArg)
{
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

        if (oldTypeArg) oldType.insert(oldType.end(), oldTypeArg->begin(), oldTypeArg->end());
        if (newTypeArg) newType.insert(newType.end(), newTypeArg->begin(), newTypeArg->end());

        registerShader(compute, "AccelerationDataStructTraverse.shader", &oldType, &newType);
        intersectRayKernels[i][r][h] = programs.back().createKernel("intersectRays");
      }
    }
  }
}

void AccelerationDataStruct::initializeData()
{
  leafNodeBoundingBoxes.create(compute);

  accXABComputeUtilId = ComputeUtil::getXABUtil(compute);
  sortComputeUtilId   = ComputeUtil::getUIntUtil(compute);

  primitiveCount  = 0;
  vertexCount     = 0;

  workgroupCount.create(compute);
  workgroupCount.resize(4, false);
}

void AccelerationDataStruct::create(ComputeInterface* compute)
{
  this->compute = compute;
  initializeData();
  registerCreateShaders();
  registerTraverseShaders();
}

uint AccelerationDataStruct::getPrimCount()const
{
  return primitiveCount;
}

void AccelerationDataStruct::bindBuffers(const ComputeMemory* vertexArray, const ComputeMemory* attributeArray,
                                         const ComputeMemory* vertexAttributeArray, DeviceArray<RTSystemSettings>* systemSettings)
{
  this->pointerVertexArray = vertexArray;
  this->pointerAttributeArray = attributeArray;
  this->pointerVertexAttributeArray = vertexAttributeArray;

  if (systemSettings)
  {
    primitiveCount  = decodePrimitiveInfo(systemSettings->host()->at(0).globalOffsets[RTPrimitiveCount-1]).primitiveOffset;
    vertexCount     = decodePrimitiveInfo(systemSettings->host()->at(0).globalOffsets[RTPrimitiveCount-1]).vertexOffset;
    pointerSystemSettings = systemSettings->device();
  }

  leafNodeBoundingBoxes.resize(primitiveCount, false);
  needsRebuild = true;
}

void AccelerationDataStruct::fullBuild()
{
  if (!needsRebuild) return;

  updatePointers();

  uint primBatchSize = 8;

  {
    uint primBatchCount = mAlignBy(primitiveCount, primBatchSize);

    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, primBatchCount);

    createPrimitiveBoundingBoxes.setArg(pointerLeafNodeBoundingBoxes, 0);
    createPrimitiveBoundingBoxes.setArg(pointerVertexArray, 1);
    createPrimitiveBoundingBoxes.setArg(pointerAttributeArray, 2);
    createPrimitiveBoundingBoxes.setArg(pointerSystemSettings, 3);
    createPrimitiveBoundingBoxes.setArg(&primBatchSize, 4);
    createPrimitiveBoundingBoxes.setArg(&primitiveCount, 5);

    compute->execute(createPrimitiveBoundingBoxes, workgroupSize, workgroupCount);

#ifdef DEBUG_ACCELERATION_DATA_STRUCT
    boundingBoxes.syncHost();
    compute->sync();
#endif
  }

  needsRebuild = false;
}

void AccelerationDataStruct::intersectRays(ComputeMemory* hits, HitStructType hitType, const ComputeMemory* rays, RayStructType rayType,
                                           uint rayCount, IntersectionType intersectionType)
{
  validateBuild();
  {
    ComputeKernel& intersectionKernel = intersectRayKernels[intersectionType][rayType][hitType];

    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, rayCount);

    intersectionKernel.setArg(hits, 0);
    intersectionKernel.setArg(rays, 1);
    intersectionKernel.setArg(&rayCount, 2);
    intersectionKernel.setArg(pointerLeafNodeBoundingBoxes, 3);
    intersectionKernel.setArg(pointerVertexArray, 4);
    intersectionKernel.setArg(pointerAttributeArray, 5);
    intersectionKernel.setArg(pointerVertexAttributeArray, 6);
    intersectionKernel.setArg(&primitiveCount, 7);
    intersectionKernel.setArg(pointerSystemSettings, 8);

    compute->execute(intersectionKernel, workgroupSize, workgroupCount);

#ifdef DEBUG_ACCELERATION_DATA_STRUCT
    boundingBoxes.syncHost();
    compute->sync();
#endif
  }
}

void AccelerationDataStruct::intersectRays(ComputeMemory* hits, HitStructType hitType, const ComputeMemory* rays, RayStructType rayType,
                                           const ComputeMemory* rayCount, IntersectionType intersectionType)
{
  validateBuild();
  {
    size_t workgroupSize[3] = {compute->maxThreadsPerGroup(), 1, 1};
    ComputeUtil::get(sortComputeUtilId)->configureWorkgroupCount(compute, workgroupCount.device(), rayCount, workgroupSize);

    ComputeKernel& intersectionKernel = intersectRayKernels[intersectionType][rayType][hitType];

    intersectionKernel.setArg(hits, 0);
    intersectionKernel.setArg(rays, 1);
    intersectionKernel.setArg(rayCount, 2);
    intersectionKernel.setArg(pointerLeafNodeBoundingBoxes, 3);
    intersectionKernel.setArg(pointerVertexArray, 4);
    intersectionKernel.setArg(pointerAttributeArray, 5);
    intersectionKernel.setArg(pointerVertexAttributeArray, 6);
    intersectionKernel.setArg(&primitiveCount, 7);
    intersectionKernel.setArg(pointerSystemSettings, 8);

    compute->execute(intersectionKernel, workgroupSize, workgroupCount.device(), 0);

#ifdef DEBUG_ACCELERATION_DATA_STRUCT
    boundingBoxes.syncHost();
    compute->sync();
#endif
  }
}
