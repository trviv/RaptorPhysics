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

uint AccelerationDataStruct::PrimitiveAttributeInfo::bindToShader(ComputeKernel& kernel, uint startIndex)
{
  if (primInfo.primType == PrimitiveSphere)
  {
    kernel.setArg(attributeBuffer[PrimitiveAttributePosition], startIndex);
    kernel.setArg(attributeBuffer[PrimitiveAttributeRadius], startIndex+1);
    kernel.setArg(&attributeInfo[PrimitiveAttributeRadius], startIndex+2);
    return startIndex+3;
  }
  else
  if (primInfo.primType == PrimitiveTriangle)
  {
    kernel.setArg(attributeBuffer[PrimitiveAttributePosition], startIndex);
    if (attributeInfo[PrimitiveAttributeIndex].strideIn4Bytes)
    {
      kernel.setArg(attributeBuffer[PrimitiveAttributeIndex], startIndex+1);
    }
    else
    {
      kernel.setArg(attributeBuffer[PrimitiveAttributePosition], startIndex+1);
    }
    kernel.setArg(&attributeInfo[PrimitiveAttributeIndex], startIndex+2);
    return startIndex+3;
  }

  return startIndex;
}

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
  assignMortonCode = programs[0].createKernel("assignMortonCode");

  includeFiles.push_back("RayStructs.h");
  includeFiles.push_back("HitStructs.h");

  for (int i=0; i<IntersectionTypeMax; i++)
  {
    for (int r=0; r<RayStructTypeMax; r++)
    {
      for (int h=0; h<HitStructTypeMax; h++)
      {
        const vector<string> oldType = {getIntersectionTypeName((IntersectionType)i), "RayStruct", "HitStruct"};
        const vector<string> newType = {"", getRayStructName((RayStructType)r), getHitStructName((HitStructType)h)};
        registerShader(compute, "AccelerationDataStructTraverse.shader", &oldType, &newType);
        intersectRayKernels[i][r][h] = programs.back().createKernel("intersectRays");
      }
    }
  }

  systemSettings.create(compute);
  boundingBoxes.create(compute);
  vertexArray.create(compute);
  attributeArray.create(compute);
  primitiveLeafData.create(compute);
  primitiveLeafDataSorted.create(compute);

  for (auto& i : registeredPrimitives)
  {
    i.clear();
  }

  accXABComputeUtilId = ComputeUtil::getXABUtil(compute);
  sortComputeUtilId   = ComputeUtil::getUIntUtil(compute);

  primitiveCount = 0;
  vertexCount    = 0;
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
  primInfo.primInfo.primType    = PrimitiveSphere;
  primInfo.primInfo.indexCount  = count;
  primInfo.primInfo.vertexCount = count;

  registeredPrimitives[PrimitiveSphere].push_back(primInfo);
}

void AccelerationDataStruct::registerTriangles(const ComputeMemory* primitiveBuffer, const ComputeMemory* indexBuffer, PackingInfo indexInfo, uint count)
{
  PrimitiveAttributeInfo primInfo;

  primInfo.attributeBuffer[PrimitiveAttributePosition]  = primitiveBuffer;
  primInfo.attributeInfo[PrimitiveAttributePosition]    = PackingInfo();
  primInfo.attributeBuffer[PrimitiveAttributeIndex]     = indexBuffer;
  primInfo.attributeInfo[PrimitiveAttributeIndex]       = indexInfo;
  primInfo.primInfo.primType    = PrimitiveTriangle;
  primInfo.primInfo.indexCount  = count;
  primInfo.primInfo.vertexCount = count * 3;

  registeredPrimitives[PrimitiveTriangle].push_back(primInfo);
}

void AccelerationDataStruct::commit()
{
  systemSettings.host()->resize(1);

  uint indexOffset  = 0;
  uint vertexOffset = 0;

  for (uint i=0; i<RTPrimitiveCount; i++)
  {
    for (const auto& p : registeredPrimitives[i])
    {
      indexOffset  += p.primInfo.indexOffset;
      vertexOffset += p.primInfo.vertexOffset;
    }

    EncodedPrimitiveInfo primInfo;

    setPrimitiveType(primInfo,         (RTPrimitiveType)i);
    setPrimitiveIndexOffset(primInfo,  indexOffset);
    setPrimitiveVertexOffset(primInfo, vertexOffset);
    systemSettings.host()->at(0).globalOffsets[i] = primInfo;
  }

  primitiveCount = indexOffset;
  vertexCount    = vertexOffset;

  boundingBoxes.resize(primitiveCount, false);
  vertexArray.resize(vertexCount, false);
  attributeArray.resize(primitiveCount, false);
  primitiveLeafData.resize(primitiveCount, false);
  primitiveLeafDataSorted.resize(primitiveCount, false);

  systemSettings.syncDevice();
}

void AccelerationDataStruct::fullBuild()
{
  uint indexOffset = 0;
  uint vertexOffset = 0;
  uint primBatchSize = 8;

  for (auto& rp : registeredPrimitives)
  {
    for (uint i=0; i<rp.size(); i++)
    {
      auto& prim = rp[i];
      uint primBatchCount = mAlignBy(prim.primInfo.indexCount, primBatchSize);
      uint primType = prim.primInfo.primType;

      size_t workgroupSize[3], workgroupCount[3];
      compute->configureSize(workgroupSize, workgroupCount, primBatchCount);

      createPrimitiveBoundingBoxes.setArg(vertexArray.device(), 0);
      createPrimitiveBoundingBoxes.setArg(attributeArray.device(), 1);
      createPrimitiveBoundingBoxes.setArg(boundingBoxes.device(), 2);
      uint nextBindIndex = prim.bindToShader(createPrimitiveBoundingBoxes, 3);
      createPrimitiveBoundingBoxes.setArg(&primBatchSize, nextBindIndex);
      createPrimitiveBoundingBoxes.setArg(&prim.primInfo.indexCount, nextBindIndex+1);
      createPrimitiveBoundingBoxes.setArg(&primType, nextBindIndex+2);
      createPrimitiveBoundingBoxes.setArg(&indexOffset, nextBindIndex+3);
      createPrimitiveBoundingBoxes.setArg(&vertexOffset, nextBindIndex+4);

      compute->execute(createPrimitiveBoundingBoxes, workgroupSize, workgroupCount);

#ifdef DEBUG_ACCELERATION_DATA_STRUCT
      boundingBoxes.syncHost();
      compute->sync();
#endif

      indexOffset += prim.primInfo.indexCount;
      vertexOffset += prim.primInfo.vertexCount;
    }
  }

  // find bounding box for the simulation space
  ComputeUtil::get(accXABComputeUtilId)->sum1D(compute, systemSettings.device(), boundingBoxes.device(), primitiveCount);

#ifdef DEBUG_ACCELERATION_DATA_STRUCT
  systemSettings.syncHost();
  compute->sync();
#endif

  {
    uint primBatchCount = mAlignBy(primitiveCount, primBatchSize);
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, primBatchCount);

    // assign morton code to the particle bounding boxes
    ComputeMemory* buffers[] = {
      primitiveLeafData.device(),
      vertexArray.device(),
      systemSettings.device()
    };
    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    assignMortonCode.setArgs(buffers, bufferCount);
    assignMortonCode.setArg(&primBatchSize, bufferCount);
    assignMortonCode.setArg(&primitiveCount, bufferCount+1);

    compute->execute(assignMortonCode, workgroupSize, workgroupCount);
  }

#ifdef DEBUG_ACCELERATION_DATA_STRUCT
  primitiveLeafData.syncHost();
  compute->sync();
#endif

  ComputeUtil::get(sortComputeUtilId)->radixSort32Bit(compute, primitiveLeafDataSorted.device(), primitiveLeafData.device(), primitiveCount);

#ifdef DEBUG_ACCELERATION_DATA_STRUCT
  primitiveLeafDataSorted.syncHost();
  compute->sync();
#endif
}

void AccelerationDataStruct::intersectRays(ComputeMemory* hits, HitStructType hitType, ComputeMemory* rays, RayStructType rayType,
                                           uint rayCount, IntersectionType intersectionType, bool initializeHit)
{
  {
    ComputeKernel& intersectionKernel = intersectRayKernels[intersectionType][rayType][hitType];

    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, rayCount);

    intersectionKernel.setArg(hits, 0);
    intersectionKernel.setArg(rays, 1);
    intersectionKernel.setArg(&rayCount, 2);
    intersectionKernel.setArg(boundingBoxes.device(),  3);
    intersectionKernel.setArg(vertexArray.device(),    4);
    intersectionKernel.setArg(attributeArray.device(), 5);
    intersectionKernel.setArg(&primitiveCount,         6);
    intersectionKernel.setArg(systemSettings.device(), 7);
    ushort initHit = initializeHit;
    intersectionKernel.setArg(&initHit,                8);

    compute->execute(intersectionKernel, workgroupSize, workgroupCount);

#ifdef DEBUG_ACCELERATION_DATA_STRUCT
    boundingBoxes.syncHost();
    compute->sync();
#endif
  }
}
