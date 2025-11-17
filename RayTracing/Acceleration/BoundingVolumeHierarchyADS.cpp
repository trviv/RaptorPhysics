/*
 * Copyright (c) Vivek Trivedi, 2025
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#include "BoundingVolumeHierarchyADS.h"

//#define DEBUG_BVH_ADS

BoundingVolumeHierarchyADS::BoundingVolumeHierarchyADS(CreationMethod treeCreationMethod)
  :maxBVHLeafs(5),sharedMemoryStride(5),bvhPersistentMultiplier(1),treeCreationMethod(treeCreationMethod)
{
  switch (treeCreationMethod)
  {
    case MaximizingParallelism:
      treeDataPtr = new MaximizingParallelismTreeData();
      break;
    case LocallyOrderedClustering:
      treeDataPtr = new LocallyOrderedClusteringTreeData();
      break;
  }
}

BoundingVolumeHierarchyADS::~BoundingVolumeHierarchyADS()
{
}

void BoundingVolumeHierarchyADS::initializeData()
{
  AccelerationDataStruct::initializeData();

  switch (treeCreationMethod)
  {
    case MaximizingParallelism:
      initializeDataMaxParallelTree();
      break;
    case LocallyOrderedClustering:
      initializeDataLocallyOrderedTree();
      break;
  }

  leafParentNodeIndices.create(compute);
  nodeParentNodeIndices.create(compute);
  treeNodeBoundingBoxes.create(compute);
  treeInternalNodes.create(compute);
  primitiveLeafData.create(compute);
  primitiveLeafDataSorted.create(compute);
  rayCounter.create(compute);

  vertexArray.create(compute);
  attributeArray.create(compute);
  vertexAttributeArray.create(compute);
  systemSettings.create(compute);

  for (auto i : primitiveInstancesPerType)
  {
    i.clear();
  }
}

void BoundingVolumeHierarchyADS::updatePointers()
{
  pointerLeafParentNodeIndices = leafParentNodeIndices.device();
  pointerNodeParentNodeIndices = nodeParentNodeIndices.device();
  pointerLeafNodeBoundingBoxes = leafNodeBoundingBoxes.device();
  pointerTreeNodeBoundingBoxes = treeNodeBoundingBoxes.device();
  pointerTreeInternalNodes     = treeInternalNodes.device();
}

void BoundingVolumeHierarchyADS::createLeafBoundingBoxes()
{
  const uint primBatchSize = 8;
  const uint primBatchCount = mAlignBy(primitiveCount, primBatchSize);

  size_t workgroupSize[3], workgroupCount[3];
  compute->configureSize(workgroupSize, workgroupCount, primBatchCount);

  createPrimitiveBoundingBoxes.setArg(pointerLeafNodeBoundingBoxes, 0);
  createPrimitiveBoundingBoxes.setArg(pointerVertexArray, 1);
  createPrimitiveBoundingBoxes.setArg(pointerAttributeArray, 2);
  createPrimitiveBoundingBoxes.setArg(pointerSystemSettings, 3);
  createPrimitiveBoundingBoxes.setArg(&primBatchSize, 4);
  createPrimitiveBoundingBoxes.setArg(&primitiveCount, 5);

  compute->execute(createPrimitiveBoundingBoxes, workgroupSize, workgroupCount);

#ifdef DEBUG_BVH_ADS
  leafNodeBoundingBoxes.syncHost();
  compute->sync();
#endif
}

void BoundingVolumeHierarchyADS::assignLeafMortonCode()
{
  const uint primBatchSize = 8;
  const uint primBatchCount = mAlignBy(primitiveCount, primBatchSize);

  size_t workgroupSize[3], workgroupCount[3];
  compute->configureSize(workgroupSize, workgroupCount, primBatchCount);

  // assign morton code to the particle bounding boxes
  assignMortonCode.setArg(primitiveLeafData.device(), 0);
  assignMortonCode.setArg(pointerVertexArray, 1);
  assignMortonCode.setArg(pointerAttributeArray, 2);
  assignMortonCode.setArg(pointerSystemSettings, 3);
  assignMortonCode.setArg(&primBatchSize, 4);
  assignMortonCode.setArg(&primitiveCount, 5);

  compute->execute(assignMortonCode, workgroupSize, workgroupCount);

#ifdef DEBUG_BVH_ADS
  primitiveLeafData.syncHost();
  compute->sync();
#endif
}

void BoundingVolumeHierarchyADS::constructTree()
{
  switch (treeCreationMethod)
  {
    case MaximizingParallelism:
      constructMaxParallelTree();
      break;
    case LocallyOrderedClustering:
      constructLocallyOrderedTree();
      break;
  }
}

void BoundingVolumeHierarchyADS::registerCreateShaders(const vector<string>* oldType, const vector<string>* newType)
{
  switch (treeCreationMethod)
  {
    case MaximizingParallelism:
    {
      MaximizingParallelismTreeData* treePtr = (MaximizingParallelismTreeData*)treeDataPtr;
      registerShader(compute, "BoundingVolumeHierarchyCreateMaxParallel.shader", oldType, newType);
      treePtr->constructBinaryTree      = programs.back().createKernel("constructBinaryTreeMaximizingParallelism");
      treePtr->constructTreeBoundingBox = programs.back().createKernel("constructTreeBoundingBoxMaximizingParallelism");
      break;
    }
    case LocallyOrderedClustering:
    {
      LocallyOrderedClusteringTreeData* treePtr = (LocallyOrderedClusteringTreeData*)treeDataPtr;
      registerShader(compute, "BoundingVolumeHierarchyCreateLocallyOrdered.shader", oldType, newType);
      treePtr->createLeafClusters = programs.back().createKernel("createLeafClustersLocallyOrdered");
      treePtr->findNearestCluster = programs.back().createKernel("findNearestClusterLocallyOrdered");
      treePtr->mergeClusters      = programs.back().createKernel("mergeClustersLocallyOrdered");
      treePtr->compactClusters    = programs.back().createKernel("compactClustersLocallyOrdered");
      break;
    }
  }

  collectPrimitives            = programs.back().createKernel("collectPrimitives");
  createPrimitiveBoundingBoxes = programs.back().createKernel("createPrimitiveBoundingBoxes");
  assignMortonCode             = programs.back().createKernel("assignMortonCode");
}

void BoundingVolumeHierarchyADS::registerTraverseShaders(const vector<string>* oldTypeArg, const vector<string>* newTypeArg)
{
  string intersectFunctionName  = "intersectRaysBVH";
  string traverseShaderProgram  = "BoundingVolumeHierarchyADSTraverse.shader";
  string traverseSharedElements = "67";

  if (oldTypeArg)
  {
    auto functionIter = find(oldTypeArg->begin(), oldTypeArg->end(), string("BVH_ADS_INTERSECT_RAY_BVH_FUNCTION"));
    if (functionIter != oldTypeArg->end())  intersectFunctionName = newTypeArg->at(functionIter - oldTypeArg->begin());

    functionIter = find(oldTypeArg->begin(), oldTypeArg->end(), string("ADS_TRAVERSAL_SHADER_PROGRAM"));
    if (functionIter != oldTypeArg->end())  traverseShaderProgram = newTypeArg->at(functionIter - oldTypeArg->begin());

    functionIter = find(oldTypeArg->begin(), oldTypeArg->end(), string("BVH_TRAVERSAL_SHARED_ELEMENTS"));
    if (functionIter != oldTypeArg->end())  traverseSharedElements = newTypeArg->at(functionIter - oldTypeArg->begin());
  }

  for (int i=0; i<IntersectionTypeMax; i++)
  {
    for (int r=0; r<RayStructTypeMax; r++)
    {
      for (int h=0; h<HitStructTypeMax; h++)
      {
        vector<string> oldType = {getIntersectionTypeName((IntersectionType)i), "RayStruct", "HitStruct"};
        vector<string> newType = {"", getRayStructName((RayStructType)r), getHitStructName((HitStructType)h)};
        appendTraversalSettings(oldType, newType);

        getRayStructDefines(oldType, newType, (RayStructType)r);
        getHitStructDefines(oldType, newType, (HitStructType)h);

        if (oldTypeArg) oldType.insert(oldType.end(), oldTypeArg->begin(), oldTypeArg->end());
        if (newTypeArg) newType.insert(newType.end(), newTypeArg->begin(), newTypeArg->end());

        registerShader(compute, traverseShaderProgram.c_str(), &oldType, &newType);
        intersectRayKernels[i][r][h] = programs.back().createKernel(intersectFunctionName.c_str());
      }
    }
  }
}

void BoundingVolumeHierarchyADS::registerResources(ComputeKernel& kernel)const
{
  kernel.registerResource(pointerTreeInternalNodes);
  kernel.registerResource(pointerLeafParentNodeIndices);
  kernel.registerResource(pointerNodeParentNodeIndices);
  kernel.registerResource(pointerLeafNodeBoundingBoxes);
  kernel.registerResource(pointerTreeNodeBoundingBoxes);
  kernel.registerResource(pointerVertexArray);
  kernel.registerResource(pointerAttributeArray);
  kernel.registerResource(pointerVertexAttributeArray);
  kernel.registerResource(pointerSystemSettings);
}

void BoundingVolumeHierarchyADS::bindBuffers(const ComputeMemory* vertexArray, const ComputeMemory* attributeArray,
                                             const ComputeMemory* vertexAttributeArray, DeviceArray<RTSystemSettings>* systemSettings)
{
  AccelerationDataStruct::bindBuffers(vertexArray, attributeArray, vertexAttributeArray, systemSettings);
  leafParentNodeIndices.resize(primitiveCount, false);
  nodeParentNodeIndices.resize(primitiveCount, false);
  treeNodeBoundingBoxes.resize(primitiveCount - 1, false);
  treeInternalNodes.resize(primitiveCount - 1, false);
  primitiveLeafData.resize(primitiveCount, false);
  primitiveLeafDataSorted.resize(primitiveCount, false);
  rayCounter.resize(1, false);

  switch (treeCreationMethod)
  {
    case MaximizingParallelism:
      resizeBuffersMaxParallelTree();
      break;
    case LocallyOrderedClustering:
      resizeBuffersLocallyOrderedTree();
      break;
  }
}

void BoundingVolumeHierarchyADS::resizePrimitiveArray()
{
  systemSettings.host()->resize(1);

  uint primOffset   = 0;
  uint vertexOffset = 0;

  for (uint i=0; i<RTPrimitiveCount; i++)
  {
    for (const auto& primInstance : primitiveInstancesPerType[i])
    {
      primOffset    += primInstance->getPrimitiveCount();
      vertexOffset  += primInstance->getPrimitiveVertexCount();
    }

    EncodedPrimitiveInfo primInfo;

    setPrimitiveType(primInfo,         (RTPrimitiveType)i);
    setPrimitiveIndexOffset(primInfo,  primOffset);
    setPrimitiveVertexOffset(primInfo, vertexOffset);
    systemSettings.host()->at(0).globalOffsets[i] = primInfo;
  }

  vertexArray.resize(vertexOffset, false);
  attributeArray.resize(primOffset, false);
  vertexAttributeArray.resize(vertexOffset, false);

  systemSettings.syncDevice();
}

void BoundingVolumeHierarchyADS::composePrimitiveArray()
{
  uint vertexOffset   = 0;
  uint primOffset     = 0;
  uint primBatchSize  = 8;

  for (const auto& primInstances : primitiveInstancesPerType)
  {
    for (const auto& primInstance : primInstances)
    {
      const auto& prim = *primInstance;
      uint primBatchCount = mAlignBy(prim.getPrimitiveCount(), primBatchSize);
      uint primType = prim.getPrimitiveType();

      size_t workgroupSize[3], workgroupCount[3];
      compute->configureSize(workgroupSize, workgroupCount, primBatchCount);

      collectPrimitives.setArg(vertexArray.device(), 0);
      collectPrimitives.setArg(attributeArray.device(), 1);
      collectPrimitives.setArg(vertexAttributeArray.device(), 2);
      uint nextBindIndex = prim.bindToShader(collectPrimitives, 3);
      collectPrimitives.setArg(&prim.getMaterialId(), nextBindIndex);
      collectPrimitives.setArg(&primBatchSize, nextBindIndex+1);
      collectPrimitives.setArg(&prim.getPrimitiveCount(), nextBindIndex+2);
      collectPrimitives.setArg(&primType, nextBindIndex+3);
      collectPrimitives.setArg(&primOffset, nextBindIndex+4);
      collectPrimitives.setArg(&vertexOffset, nextBindIndex+5);
      collectPrimitives.setArg(&prim.getTransform(), nextBindIndex+6);

      compute->execute(collectPrimitives, workgroupSize, workgroupCount);

#ifdef DEBUG_BVH_ADS
      vertexArray.syncHost();
      attributeArray.syncHost();
      vertexAttributeArray.syncHost();
      compute->sync();
#endif

      vertexOffset  += prim.getPrimitiveVertexCount();
      primOffset    += prim.getPrimitiveCount();
    }
  }
}

void BoundingVolumeHierarchyADS::fullBuild()
{
  if (!needsRebuild) return;

  updatePointers();

  createLeafBoundingBoxes();

  // find bounding box for the simulation space
  ComputeUtil::get(accXABComputeUtilId)->sum1D(compute, pointerSystemSettings, pointerLeafNodeBoundingBoxes, primitiveCount);

#ifdef DEBUG_BVH_ADS
  systemSettings.syncHost();
  compute->sync();
#endif

  assignLeafMortonCode();

  ComputeUtil::get(sortComputeUtilId)->radixSort32Bit(compute, primitiveLeafDataSorted.device(), primitiveLeafData.device(), primitiveCount);

#ifdef DEBUG_BVH_ADS
  primitiveLeafDataSorted.syncHost();
  compute->sync();
#endif

  constructTree();

  needsRebuild = false;
}

void BoundingVolumeHierarchyADS::intersectRays(ComputeMemory* hits, HitStructType hitType, const ComputeMemory* rays, RayStructType rayType,
                                               uint rayCount, IntersectionType intersectionType)
{
  validateBuild();
  {
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, mAlignBy(rayCount, bvhPersistentMultiplier));
    if (bvhPersistentMultiplier > 1)
    {
      ComputeUtil::get(sortComputeUtilId)->clearBuffer(compute, rayCounter.device(), 1);
    }

    ComputeKernel& intersectionKernel = intersectRayKernels[intersectionType][rayType][hitType];

    intersectionKernel.setArg(hits, 0);
    intersectionKernel.setArg(rays, 1);
    intersectionKernel.setArg(&rayCount, 2);
    intersectionKernel.setArg(pointerVertexArray, 3);
    intersectionKernel.setArg(pointerAttributeArray, 4);
    intersectionKernel.setArg(pointerVertexAttributeArray, 5);
    intersectionKernel.setArg(pointerTreeInternalNodes, 6);
    intersectionKernel.setArg(pointerLeafParentNodeIndices, 7);
    intersectionKernel.setArg(pointerNodeParentNodeIndices, 8);
    intersectionKernel.setArg(pointerLeafNodeBoundingBoxes, 9);
    intersectionKernel.setArg(pointerTreeNodeBoundingBoxes, 10);
    intersectionKernel.setArg(pointerSystemSettings, 11);
    intersectionKernel.setArg(&primitiveCount, 12);
    intersectionKernel.setArg(rayCounter.device(), 13);
    intersectionKernel.setSharedMemArg(4 * max(workgroupSize[0] * workgroupSize[1] * workgroupSize[2] * sharedMemoryStride, (size_t)4), 14);

    compute->execute(intersectionKernel, workgroupSize, workgroupCount);

#ifdef DEBUG_BVH_ADS
    leafNodeBoundingBoxes.syncHost();
    compute->sync();
#endif
  }
}

void BoundingVolumeHierarchyADS::intersectRays(ComputeMemory* hits, HitStructType hitType, const ComputeMemory* rays, RayStructType rayType,
                                               const ComputeMemory* rayCount, IntersectionType intersectionType)
{
  validateBuild();
  {
    size_t workgroupSize[3] = {compute->maxThreadsPerGroup() * bvhPersistentMultiplier, 1, 1};
    ComputeUtil::get(sortComputeUtilId)->configureWorkgroupCount(compute, workgroupCount.device(), rayCount, workgroupSize);
    workgroupSize[0] /= bvhPersistentMultiplier;

    ComputeKernel& intersectionKernel = intersectRayKernels[intersectionType][rayType][hitType];

    intersectionKernel.setArg(hits, 0);
    intersectionKernel.setArg(rays, 1);
    intersectionKernel.setArg(rayCount, 2);
    intersectionKernel.setArg(pointerVertexArray, 3);
    intersectionKernel.setArg(pointerAttributeArray, 4);
    intersectionKernel.setArg(pointerVertexAttributeArray, 5);
    intersectionKernel.setArg(pointerTreeInternalNodes, 6);
    intersectionKernel.setArg(pointerLeafParentNodeIndices, 7);
    intersectionKernel.setArg(pointerNodeParentNodeIndices, 8);
    intersectionKernel.setArg(pointerLeafNodeBoundingBoxes, 9);
    intersectionKernel.setArg(pointerTreeNodeBoundingBoxes, 10);
    intersectionKernel.setArg(pointerSystemSettings, 11);
    intersectionKernel.setArg(&primitiveCount, 12);
    intersectionKernel.setArg(workgroupCount.device(), 13);
    intersectionKernel.setSharedMemArg(4 * max(workgroupSize[0] * workgroupSize[1] * workgroupSize[2] * sharedMemoryStride, (size_t)4), 14);

    compute->execute(intersectionKernel, workgroupSize, workgroupCount.device(), 0);

#ifdef DEBUG_BVH_ADS
    leafNodeBoundingBoxes.syncHost();
    compute->sync();
#endif
  }
}

void BoundingVolumeHierarchyADS::appendTraversalSettings(vector<string>& oldType, vector<string>& newType)const
{
  AccelerationDataStruct::appendTraversalSettings(oldType, newType);

  oldType.insert(oldType.end(), {"BVH_ADS_PERSISTENT_MULTIPLIER", "RAY_TRAVERSAL_BVH_MAX_LEAFS", "RAY_TRAVERSAL_SHARED_MEMORY_INDEX_STRIDE", "BVH_TRAVERSAL_SHARED_ELEMENTS"});
  newType.insert(newType.end(), {to_string(bvhPersistentMultiplier), to_string(maxBVHLeafs), to_string(sharedMemoryStride), "67"});
//  oldType.push_back("STACKLESS_TRAVERSE_EARLY_CHILD");
//  newType.push_back("");
  oldType.push_back("BVH_STACK_TRAVERSAL_CACHE_LAST_NODE");
  newType.push_back("");
}
