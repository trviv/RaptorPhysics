/*
 * Copyright (c) Vivek Trivedi, 2025
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#include "BoundingVolumeHierarchyADS.h"

//#define DEBUG_BVH_ADS

void BoundingVolumeHierarchyADS::initializeDataLocallyOrderedTree()
{
  LocallyOrderedClusteringTreeData* treePtr = (LocallyOrderedClusteringTreeData*)treeDataPtr;

  treePtr->clusterValidFlags.create(compute);
  treePtr->clusterNewFlags.create(compute);
  treePtr->nearestNeighbourIndex.create(compute);
  treePtr->clusterCounters.create(compute, NULL, true);
  treePtr->clusterBoundingBoxes[0].create(compute);
  treePtr->clusterBoundingBoxes[1].create(compute);
  treePtr->clusterTempInternalNodes.create(compute);
  treePtr->clusterNodes[0].create(compute);
  treePtr->clusterNodes[1].create(compute);
}

void BoundingVolumeHierarchyADS::constructLocallyOrderedTree()
{
  LocallyOrderedClusteringTreeData* treePtr = (LocallyOrderedClusteringTreeData*)treeDataPtr;

  ComputeMemory clusterCount[2]         = {ComputeMemory(treePtr->clusterCounters.device(), 0), ComputeMemory(treePtr->clusterCounters.device(), 4*8)};
  ComputeMemory clusterDispatchSize[2]  = {ComputeMemory(treePtr->clusterCounters.device(), 4*16), ComputeMemory(treePtr->clusterCounters.device(), 4*24)};
  ComputeMemory internalNodeCount[2]    = {ComputeMemory(treePtr->clusterCounters.device(), 4*32), ComputeMemory(treePtr->clusterCounters.device(), 4*40)};

  std::fill(treePtr->clusterCounters.host()->begin(), treePtr->clusterCounters.host()->end(), 0);
  (*treePtr->clusterCounters.host())[0] = primitiveCount;
  (*treePtr->clusterCounters.host())[1] = 1;
  (*treePtr->clusterCounters.host())[2] = 1;
  (*treePtr->clusterCounters.host())[8] = primitiveCount;
  (*treePtr->clusterCounters.host())[9] = 1;
  (*treePtr->clusterCounters.host())[10] = 1;
  treePtr->clusterCounters.syncDevice();

  const int searchRadius  = 512;

  {
    size_t workgroupSize[3], workgroupCount[3];
    compute->configureSize(workgroupSize, workgroupCount, primitiveCount);

    treePtr->createLeafClusters.setArg(treePtr->clusterNodes[0].device(), 0);
    treePtr->createLeafClusters.setArg(leafNodeBoundingBoxes.device(), 1);
    treePtr->createLeafClusters.setArg(&clusterCount[0], 2);

    compute->execute(treePtr->createLeafClusters, workgroupSize, workgroupCount);

#ifdef DEBUG_BVH_ADS
    treePtr->clusterNodes[0].syncHost();
    compute->sync();
#endif

    compute->copyBuffer(leafNodeBoundingBoxes.device(), treePtr->clusterBoundingBoxes[0].device(), 0, 0, sizeof(XAB) * primitiveCount);
  }

  size_t workgroupSize[3] = {compute->maxThreadsPerGroup(), 1, 1};

  int i = 0;
  for (; treePtr->clusterCounters.host()->at((i&1) * 8) > 1; i++)
  {
    ComputeUtil::get(sortComputeUtilId)->configureWorkgroupCount(compute, &clusterDispatchSize[i & 1], &clusterCount[i & 1], workgroupSize);

    treePtr->findNearestCluster.setArg(treePtr->nearestNeighbourIndex.device(), 0);
    treePtr->findNearestCluster.setArg(treePtr->clusterNodes[i & 1].device(), 1);
    treePtr->findNearestCluster.setArg(&clusterCount[i & 1], 2);
    treePtr->findNearestCluster.setArg(&searchRadius, 3);

    compute->execute(treePtr->findNearestCluster, workgroupSize, &clusterDispatchSize[i & 1], 0);

#ifdef DEBUG_BVH_ADS
    treePtr->nearestNeighbourIndex.syncHost();
    treePtr->clusterCounters.syncHost();
    compute->sync();
#endif

    treePtr->mergeClusters.setArg(treePtr->clusterValidFlags.device(), 0);
    treePtr->mergeClusters.setArg(treePtr->clusterNewFlags.device(), 1);
    treePtr->mergeClusters.setArg(treePtr->clusterNodes[i & 1].device(), 2);
    treePtr->mergeClusters.setArg(treePtr->clusterBoundingBoxes[i & 1].device(), 3);
    treePtr->mergeClusters.setArg(treePtr->clusterTempInternalNodes.device(), 4);
    treePtr->mergeClusters.setArg(treePtr->nearestNeighbourIndex.device(), 5);
    treePtr->mergeClusters.setArg(&clusterCount[i & 1], 6);

    compute->execute(treePtr->mergeClusters, workgroupSize, &clusterDispatchSize[i & 1], 0);

#ifdef DEBUG_BVH_ADS
    treePtr->clusterValidFlags.syncHost();
    treePtr->clusterNewFlags.syncHost();
    treePtr->nearestNeighbourIndex.syncHost();
    treePtr->clusterNodes[i & 1].syncHost();
    treePtr->clusterBoundingBoxes[i & 1].syncHost();
    treePtr->clusterTempInternalNodes.syncHost();
    compute->sync();
#endif

    ComputeUtil::get(sortComputeUtilId)->prefixScan1D(compute, treePtr->clusterValidFlags.device(), treePtr->clusterValidFlags.device(), &clusterCount[i & 1], primitiveCount);
    ComputeUtil::get(sortComputeUtilId)->prefixScan1D(compute, treePtr->clusterNewFlags.device(), treePtr->clusterNewFlags.device(), &clusterCount[i & 1], primitiveCount);

#ifdef DEBUG_BVH_ADS
    treePtr->clusterValidFlags.syncHost();
    treePtr->clusterNewFlags.syncHost();
    compute->sync();
#endif

    ComputeMemory* buffers[] = {
      treePtr->clusterBoundingBoxes[(i & 1) ^ 1].device(),
      treePtr->clusterNodes[(i & 1) ^ 1].device(),
      treeInternalNodes.device(),
      &internalNodeCount[(i & 1) ^ 1],
      &clusterCount[(i & 1) ^ 1],
      leafParentNodeIndices.device(),
      nodeParentNodeIndices.device(),
      treeNodeBoundingBoxes.device(),

      treePtr->clusterBoundingBoxes[i & 1].device(),
      treePtr->clusterNodes[i & 1].device(),
      treePtr->clusterTempInternalNodes.device(),
      &internalNodeCount[i & 1],
      &clusterCount[i & 1],
      treePtr->clusterValidFlags.device(),
      treePtr->clusterNewFlags.device()
    };

    uint bufferCount = sizeof(buffers) / sizeof(ComputeMemory*);
    treePtr->compactClusters.setArgs(buffers, bufferCount);
    treePtr->compactClusters.setArg(&primitiveCount, bufferCount);

    compute->execute(treePtr->compactClusters, workgroupSize, &clusterDispatchSize[i & 1], 0);

#ifdef DEBUG_BVH_ADS
    treePtr->clusterNodes[(i & 1) ^ 1].syncHost();
    treePtr->clusterNodes[i & 1].syncHost();
    treePtr->clusterCounters.syncHost();
    treePtr->clusterBoundingBoxes[(i & 1) ^ 1].syncHost();
    treeInternalNodes.syncHost();
    leafParentNodeIndices.syncHost();
    nodeParentNodeIndices.syncHost();
    compute->sync();
#endif

    treePtr->clusterCounters.syncHost();
    compute->sync();
  }

  logComputeMessage("Parallel locally ordered clustering completed in %d iterations", i);
}

void BoundingVolumeHierarchyADS::resizeBuffersLocallyOrderedTree()
{
  LocallyOrderedClusteringTreeData* treePtr = (LocallyOrderedClusteringTreeData*)treeDataPtr;

  treePtr->clusterValidFlags.resize(primitiveCount, false);
  treePtr->clusterNewFlags.resize(primitiveCount, false);
  treePtr->nearestNeighbourIndex.resize(primitiveCount, false);
  treePtr->clusterCounters.resize(8*6, false);
  treePtr->clusterBoundingBoxes[0].resize(primitiveCount, false);
  treePtr->clusterBoundingBoxes[1].resize(primitiveCount, false);
  treePtr->clusterTempInternalNodes.resize(primitiveCount, false);
  treePtr->clusterNodes[0].resize(primitiveCount, false);
  treePtr->clusterNodes[1].resize(primitiveCount, false);

  treePtr->clusterCounters.host()->resize(8*6);
}
