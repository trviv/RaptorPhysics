#ifndef BOUNDING_VOLUME_HIERARCHY_ADS_H
#define BOUNDING_VOLUME_HIERARCHY_ADS_H

#include "AccelerationDataStruct.h"

struct MaximizingParallelismTreeData
{
  ComputeKernel collectPrimitives;
  ComputeKernel assignMortonCode;
  ComputeKernel constructBinaryTree;
  ComputeKernel constructTreeBoundingBox;

  DeviceArray<uint> visitedInternalNodes;
};

struct LocallyOrderedClusteringTreeData
{
  ComputeKernel createLeafClusters;
  ComputeKernel findNearestCluster;
  ComputeKernel mergeClusters;
  ComputeKernel compactClusters;

  DeviceArray<int> clusterValidFlags;
  DeviceArray<int> clusterNewFlags;
  DeviceArray<int> nearestNeighbourIndex;
  DeviceArray<int> clusterCounters;
  DeviceArray<XAB> clusterBoundingBoxes[2];
  DeviceArray<BVHNodeInfo> clusterTempInternalNodes;
  DeviceArray<BVHClusterStruct> clusterNodes[2];
};

/*!
@class Class to solve collisions using Binary Radix Tree
@info Fast BVH Construction on GPUs C. Lauterbach and M. Garland and S. Sengupta and D. Luebke and D. Manocha.
*/
class BoundingVolumeHierarchyADS : public AccelerationDataStruct
{
  void *treeDataPtr;

  void initializeDataMaxParallelTree();
  void constructMaxParallelTree();
  void resizeBuffersMaxParallelTree();

  void initializeDataLocallyOrderedTree();
  void constructLocallyOrderedTree();
  void resizeBuffersLocallyOrderedTree();

public:
  
  enum CreationMethod
  {
    MaximizingParallelism,    // Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees 2012
    LocallyOrderedClustering, // Parallel Locally-Ordered Clustering for Bounding Volume Hierarchy Construction 2017
  };

protected:

  uint maxBVHLeafs;
  uint sharedMemoryStride;
  uint bvhPersistentMultiplier;
  CreationMethod treeCreationMethod;

  ComputeKernel collectPrimitives;
  ComputeKernel assignMortonCode;

  DeviceArray<uint>         leafParentNodeIndices;
  DeviceArray<uint>         nodeParentNodeIndices;
  DeviceArray<XAB>          treeNodeBoundingBoxes;
  DeviceArray<BVHNodeInfo>  treeInternalNodes;
  DeviceArray<BVHLeafInfo>  primitiveLeafData;
  DeviceArray<BVHLeafInfo>  primitiveLeafDataSorted;
  DeviceArray<int>          rayCounter;

  vector<const RayTracingEntity*> primitiveInstancesPerType[RTPrimitiveCount];
  /*!@member Composite array containing all positions.*/
  DeviceArray<PrimitiveStruct>    vertexArray;
  /*!@member Composite array containing all primitive attributes.*/
  DeviceArray<PrimitiveAttrib>    attributeArray;
  /*!@member Composite array containing all vertex attributes.*/
  DeviceArray<VertexAttrib>       vertexAttributeArray;
  /*!@member Primitive data for the whole system.*/
  DeviceArray<RTSystemSettings>   systemSettings;

  void createBuffers(ComputeInterface* compute);

  void initializeData();

  void updatePointers();

  virtual void createLeafBoundingBoxes();

  virtual void assignLeafMortonCode();

  void constructTree();

  /*!@function Create Acceleration Data Structure creation shaders.*/
  void registerCreateShaders(const vector<string>* oldType = NULL, const vector<string>* newType = NULL);

  /*!@function Create Acceleration Data Structure traverse shaders.*/
  void registerTraverseShaders(const vector<string>* oldType = NULL, const vector<string>* newType = NULL);

  void registerResources(ComputeKernel& kernel)const;

  void resizePrimitiveArray();

  void composePrimitiveArray();

public:

  BoundingVolumeHierarchyADS(CreationMethod treeCreationMethod);

  ~BoundingVolumeHierarchyADS();

  void bindBuffers(const ComputeMemory* vertexArray, const ComputeMemory* attributeArray,
                   const ComputeMemory* vertexAttributeArray, DeviceArray<RTSystemSettings>* systemSettings);

  void fullBuild();

  void intersectRays(ComputeMemory* hits, HitStructType hitType, const ComputeMemory* rays, RayStructType rayType,
                     uint rayCount, IntersectionType intersectionType);

  void intersectRays(ComputeMemory* hits, HitStructType hitType, const ComputeMemory* rays, RayStructType rayType,
                     const ComputeMemory* rayCount, IntersectionType intersectionType);
};

#endif
