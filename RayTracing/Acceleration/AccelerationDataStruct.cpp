#include "AccelerationDataStruct.h"

uint AccelerationDataStruct::accXABComputeUtilId = -1;

#define DEBUG_ACCELERATION_DATA_STRUCT

uint AccelerationDataStruct::PrimitiveAttributeInfo::bindToShader(ComputeKernel kernel, uint startIndex)
{
  if (type == PrimitiveSphere)
  {
    kernel.setArg(attributeBuffer[PrimitiveAttributePosition], startIndex);
    kernel.setArg(attributeBuffer[PrimitiveAttributeRadius], startIndex+1);
    kernel.setArg(&attributeInfo[PrimitiveAttributeRadius], startIndex+2);
    return startIndex+3;
  }

  return startIndex;
}

AccelerationDataStruct::AccelerationDataStruct()
{

}

string AccelerationDataStruct::createIntersectionKey(RayStructType rayType, HitStructType hitType)const
{
  return getRayStructName(rayType)+":"+getHitStructName(hitType);
}

AccelerationDataStruct::~AccelerationDataStruct()
{
  registeredPrimitives.clear();
  boundingBoxes.free();
}

void AccelerationDataStruct::create(ComputeInterface* compute)
{
  this->compute = compute;
  includeFiles.push_back("ComputeHeader.shader");
  includeFiles.push_back("ComputeShared.h");
  includeFiles.push_back("RayStructs.h");
  includeFiles.push_back("RayTracingStruct.h");
  includeFiles.push_back("HitStructs.h");

  registerShader(compute, "AccelerationDataStructCreate.shader", NULL, NULL);

  createPrimitiveBoundingBoxes = programs[0].createKernel("createPrimitiveBoundingBoxes");

  for (int r=0; r<RayStructTypeMax; r++)
  {
    for (int h=0; h<HitStructTypeMax; h++)
    {
      const vector<string> oldType = {"RayStruct", "HitStruct"};
      const vector<string> newType = {getRayStructName((RayStructType)r), getHitStructName((HitStructType)h)};
      registerShader(compute, "AccelerationDataStructCreate.shader", &oldType, &newType);
      string key = createIntersectionKey((RayStructType)r, (HitStructType)h);
      intersectRayKernels[key] = programs.back().createKernel("createPrimitiveBoundingBoxes");
//      registerShader(compute, "AccelerationDataStructTraverse.shader", &oldType, &newType);
    }
  }

  boundingBoxes.create(compute, NULL);

  map<ComputeUtilKey, string> lbvhXABSetting;
  lbvhXABSetting[ComputeUtilBatchSize] = "1";
  lbvhXABSetting[ComputeUtilStructType] = "XAB";
  lbvhXABSetting[ComputeUtilStructSize] = "32";
  lbvhXABSetting[ComputeUtilOnlyReduce] = "1";
  lbvhXABSetting[ComputeUtilCustomAddFunction] = "mergeXAB";
  lbvhXABSetting[ComputeUtilCustomDivFunction] = "divXAB";
  lbvhXABSetting[ComputeUtilCustomCopyFunction] = "copyXAB";
  lbvhXABSetting[ComputeUtilCustomClearFunction] = "clearXAB";
  lbvhXABSetting[ComputeUtilCustomReduceFunction] = "reduceXAB";
  lbvhXABSetting[ComputeUtilSkipParallelPrimitives] = "1";
  accXABComputeUtilId = ComputeUtil::create(compute, lbvhXABSetting, NULL);

  primitiveCount = 0;
}

uint AccelerationDataStruct::getPrimCount()const
{
  return primitiveCount;
}

void AccelerationDataStruct::registerSpheres(const ComputeMemory* primitiveBuffer, const ComputeMemory* radiusBuffer, PackingInfo radiusInfo, uint count)
{
  PrimitiveAttributeInfo primInfo;

  primInfo.attributeBuffer[PrimitiveAttributePosition]  = primitiveBuffer;
  primInfo.attributeInfo[PrimitiveAttributePosition]    = PackingInfo();
  primInfo.attributeBuffer[PrimitiveAttributeRadius]    = radiusBuffer;
  primInfo.attributeInfo[PrimitiveAttributeRadius]      = radiusInfo;
  primInfo.count                                        = count;
  primInfo.type                                         = PrimitiveSphere;

  registeredPrimitives.push_back(primInfo);
  primitiveCount += count;
}

void AccelerationDataStruct::fullUpdate()
{
  boundingBoxes.resize(primitiveCount, false);

  uint primitiveOffset = 0;

  // TODO: Add offsetting into group bounding box buffer
  for (auto& prim : registeredPrimitives)
  {
    uint primBatchSize = 8;
    uint primBatchCount = mAlignBy(prim.count, primBatchSize);

    // add to system bounding box
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, primBatchCount);

    createPrimitiveBoundingBoxes.setArg(boundingBoxes.device(), 0);
    uint nextBindIndex = prim.bindToShader(createPrimitiveBoundingBoxes, 1);
    createPrimitiveBoundingBoxes.setArg(&primBatchSize, nextBindIndex);
    createPrimitiveBoundingBoxes.setArg(&prim.count, nextBindIndex+1);
    createPrimitiveBoundingBoxes.setArg(&primitiveOffset, nextBindIndex+2);
    primitiveOffset += prim.count;

    compute->execute(createPrimitiveBoundingBoxes, workgroupSize, workgroupCount);

#ifdef DEBUG_ACCELERATION_DATA_STRUCT
    boundingBoxes.syncHost();
    compute->sync();
#endif
  }
}

void AccelerationDataStruct::intersectRays(ComputeMemory* hits, HitStructType hitType, ComputeMemory* rays, RayStructType rayType)
{

}
