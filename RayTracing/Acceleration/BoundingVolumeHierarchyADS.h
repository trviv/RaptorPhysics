#ifndef BOUNDING_VOLUME_HIERARCHY_ADS_H
#define BOUNDING_VOLUME_HIERARCHY_ADS_H

#include "AccelerationDataStruct.h"

/*!
@class Class to solve collisions using Binary Radix Tree
@info Fast BVH Construction on GPUs C. Lauterbach and M. Garland and S. Sengupta and D. Luebke and D. Manocha.
*/
class BoundingVolumeHierarchyADS : public AccelerationDataStruct
{
protected:

  ComputeKernel assignMortonCode;
  ComputeKernel constructBinaryTree;
  ComputeKernel constructTreeBoundingBox;

  DeviceArray<uint>         visitedInternalNodes;
  DeviceArray<uint>         leafParentNodeIndices;
  DeviceArray<uint>         nodeParentNodeIndices;
  DeviceArray<XAB>          treeNodeBoundingBoxes;
  DeviceArray<BVHNodeInfo>  treeInternalNodes;
  DeviceArray<BVHLeafInfo>  primitiveLeafData;
  DeviceArray<BVHLeafInfo>  primitiveLeafDataSorted;

  const ComputeMemory*  pointerLeafParentNodeIndices;
  const ComputeMemory*  pointerNodeParentNodeIndices;
  const ComputeMemory*  pointerTreeNodeBoundingBoxes;
  const ComputeMemory*  pointerTreeInternalNodes;

  void createBuffers(ComputeInterface* compute);

  void initializeData();

  void updatePointers();

  /*!@function Create Acceleration Data Structure creation shaders.*/
  void registerCreateShaders(const vector<string>* oldType = NULL, const vector<string>* newType = NULL);

  /*!@function Create Acceleration Data Structure traverse shaders.*/
  void registerTraverseShaders(const vector<string>* oldType = NULL, const vector<string>* newType = NULL);

public:

  BoundingVolumeHierarchyADS();

  ~BoundingVolumeHierarchyADS();

  void bindBuffers(const ComputeMemory* vertexArray, const ComputeMemory* attributeArray,
                   DeviceArray<RTSystemSettings>* systemSettings);

  void fullBuild();

  void intersectRays(ComputeMemory* hits, HitStructType hitType, const ComputeMemory* rays, RayStructType rayType,
                     uint rayCount, IntersectionType intersectionType);

  void intersectRays(ComputeMemory* hits, HitStructType hitType, const ComputeMemory* rays, RayStructType rayType,
                     const ComputeMemory* rayCount, IntersectionType intersectionType);
};

#endif
