#ifndef BOUNDING_VOLUME_HIERARCHY_ADS_TRAVERSE_SHADER_H
#define BOUNDING_VOLUME_HIERARCHY_ADS_TRAVERSE_SHADER_H

//#define BOUNDING_VOLUME_HIERARCHY_ADS_DEBUG_TREE_TRAVERSAL

#define BVH_TRAVERSAL_FROM_PARENT   1
#define BVH_TRAVERSAL_FROM_CHILD    2
#define BVH_TRAVERSAL_FROM_SIBLING  3

/*inline float3 stacklessTraverseBinaryTree(
  const Thread ParticleStruct*        currentParticle,
  const Device ParticleStruct*        particlesPredictedOld,
  const ParticleDifferential          selfParticleDiff,
  const Device ParticleDifferential*  particlesDiff,
  const Device BVHNodeInfo*           treeInternalNodes,
  const Device uint*                  leafParentNodeIndices,
  const Device uint*                  nodeParentNodeIndices,
  const Device XAB*                   treeInternalNodeBoundingBoxes,
  const Thread ParticleCollisionData* collisionData,
  const ushort                        stablizationPass,
  const short                         solverType,
  Device ParticleStruct*              particlesDelta,
#ifdef MARK_COLLIDED_PARTICLES
  Device ParticleCollisionData*       particleCollisionData,
#else
  const Device ParticleCollisionData* particleCollisionData,
#endif
  const Thread CollisionSolverData*   collisionSolverData,
  const int                           index)
{
#ifdef MARK_COLLIDED_PARTICLES
  bool collided = false;
#endif

  float3 output = constructFloat3(0.f);
  uint collisionCount = 0;

  XAB particleBoundingBox;
  particleBoundingBox.min = currentParticle->position - constructFloat3(collisionData->radius);
  particleBoundingBox.max = currentParticle->position + constructFloat3(collisionData->radius);

  const float sdfMagnitude = collisionData->gradientMagnitude;

  // mark index of the root node internal
  uint currNodeIndex = setBVHInternalNodeMarker(0, 0);
  uchar state = BVH_TRAVERSAL_FROM_PARENT;

  INIT_POLL();

  while (true && !POLL_TIMEOUT())
  {
    uint nextcurrNodeIndex;

    // traverse while a leaf node is found
    while (!POLL_TIMEOUT())
    {
      bool switchBit;
      uint parentIndex;
      BVHNodeInfo parentNode;

      if (currNodeIndex != setBVHInternalNodeMarker(0, 0))
      {
        // fetch parent index
        parentIndex = select(nodeParentNodeIndices[removeBVHInternalNodeMarker(currNodeIndex)], leafParentNodeIndices[currNodeIndex], isBVHLeafNode(currNodeIndex));

        // mark as internal node
        if (isBVHLeafNode(parentIndex))
        {
          parentIndex = setBVHInternalNodeMarker(0, parentIndex);
        }

        // parentNode is only needed if from parent or child
        if (state != BVH_TRAVERSAL_FROM_SIBLING)
        {
          parentNode = treeInternalNodes[removeBVHInternalNodeMarker(parentIndex)];
        }
      }
      else
      {
        parentIndex = BOUNDING_VOLUME_HIERARCHY_ADS_ROOT_NODE_MARKER;
      }

#ifdef BOUNDING_VOLUME_HIERARCHY_ADS_DEBUG_TREE_TRAVERSAL
      printf("Node: %d %d %d\n", index, currNodeIndex, parentIndex);
#endif

      if (state == BVH_TRAVERSAL_FROM_CHILD)
      {
        if (currNodeIndex == BOUNDING_VOLUME_HIERARCHY_ADS_ROOT_NODE_MARKER)
        {
          break;
        }

#ifdef BOUNDING_VOLUME_HIERARCHY_ADS_DEBUG_TREE_TRAVERSAL
        if (currNodeIndex == parentNode.child[0])
        {
          printf("C->S: %d %d %d\n", index, currNodeIndex, parentNode.child[1]);
          currNodeIndex = parentNode.child[1];
          state = BVH_TRAVERSAL_FROM_SIBLING;
        }
        else
        {
          printf("C->P: %d %d %d\n", index, currNodeIndex, parentIndex);
          currNodeIndex = parentIndex;
          state = BVH_TRAVERSAL_FROM_CHILD;
        }
#else
        switchBit = (currNodeIndex == parentNode.child[0]);
        currNodeIndex = select(parentIndex, parentNode.child[1], switchBit);
        state = select(BVH_TRAVERSAL_FROM_CHILD, BVH_TRAVERSAL_FROM_SIBLING, switchBit);
#endif
      }
      else // from parent or silbing
      {
        const XAB boundingBox = treeInternalNodeBoundingBoxes[select(removeBVHInternalNodeMarker(currNodeIndex), (int)currNodeIndex, isBVHLeafNode(currNodeIndex))];

#ifndef BOUNDING_VOLUME_HIERARCHY_ADS_DEBUG_TREE_TRAVERSAL
        // store the incoming state
        switchBit = (state == BVH_TRAVERSAL_FROM_SIBLING);

        // switch to next state
        state = select(BVH_TRAVERSAL_FROM_SIBLING, BVH_TRAVERSAL_FROM_CHILD, switchBit);
#endif

        // leaf test has to be done before intersect XAB so that all leaf siblings are processed else it may get skipped
        if (isBVHLeafNode(currNodeIndex))
        {
#ifdef BOUNDING_VOLUME_HIERARCHY_ADS_DEBUG_TREE_TRAVERSAL
          if (state == BVH_TRAVERSAL_FROM_SIBLING)
          {
            printf("S->C: %d %d %d\n", index, currNodeIndex, parentIndex);
            nextcurrNodeIndex = parentIndex;
            state = BVH_TRAVERSAL_FROM_CHILD;
          }
          else
          {
            printf("P->S: %d %d %d\n", index, currNodeIndex, parentNode.child[1]);
            nextcurrNodeIndex = parentNode.child[1];
            state = BVH_TRAVERSAL_FROM_SIBLING;
          }
#else
          nextcurrNodeIndex = select(parentNode.child[1], parentIndex, switchBit);
#endif
          break;
        }
        else if (intersectXAB(&particleBoundingBox, &boundingBox) == 0)
        {
#ifdef BOUNDING_VOLUME_HIERARCHY_ADS_DEBUG_TREE_TRAVERSAL
          if (state == BVH_TRAVERSAL_FROM_SIBLING)
          {
            printf("S->C: %d %d %d\n", index, currNodeIndex, parentIndex);
            currNodeIndex = parentIndex;
            state = BVH_TRAVERSAL_FROM_CHILD;
          }
          else
          {
            printf("P->S: %d %d %d\n", index, currNodeIndex, parentNode.child[1]);
            currNodeIndex = parentNode.child[1];
            state = BVH_TRAVERSAL_FROM_SIBLING;
          }
#else
          currNodeIndex = select(parentNode.child[1], parentIndex, switchBit);
#endif
        }
        else
        {
          BVHNodeInfo node = treeInternalNodes[removeBVHInternalNodeMarker(currNodeIndex)];
#ifdef BOUNDING_VOLUME_HIERARCHY_ADS_DEBUG_TREE_TRAVERSAL
          printf(" ->C: %d %d %d\n", index, currNodeIndex, node.child[0]);
#endif
          currNodeIndex = node.child[0];
          state = BVH_TRAVERSAL_FROM_PARENT;
        }
      }
    }

    if (currNodeIndex == BOUNDING_VOLUME_HIERARCHY_ADS_ROOT_NODE_MARKER)
    {
      break;
    }

#ifdef BOUNDING_VOLUME_HIERARCHY_ADS_DEBUG_TREE_TRAVERSAL
    printf("Test: %d %d\n", index, currNodeIndex);
#endif
    // while ()
    // test colision if not an invalid node
    if (index != currNodeIndex)
    {
      const ParticleStruct otherParticle = particlesPredictedOld[currNodeIndex];
      if (shouldCheckForCollision(solverType, index, currNodeIndex, currentParticle, &otherParticle, true))
      {
        const ParticleDifferential otherParticleDiff = particlesDiff[currNodeIndex];
        output += processParticleCollision(currentParticle, &selfParticleDiff, &otherParticle, &otherParticleDiff, true,
          collisionData, collisionSolverData, currNodeIndex, index, sdfMagnitude, &collisionCount, stablizationPass, solverType, particlesDelta,
#ifdef MARK_COLLIDED_PARTICLES
          particleCollisionData, &collided);
#else
          particleCollisionData);
#endif
      }
    }

    if (nextcurrNodeIndex == BOUNDING_VOLUME_HIERARCHY_ADS_ROOT_NODE_MARKER)
    {
      break;
    }

    currNodeIndex = nextcurrNodeIndex;
  }

#ifdef MARK_COLLIDED_PARTICLES
  particleCollisionData[index].radius = fabs(collisionData->radius) * (collided ? -1.f : 1.f);
#endif

  if (collisionCount)
  {
    output /= collisionCount;
  }

  return output;
}*/

inline HitStruct stackTraverseBinaryTree(
  float                         currentTime,
  const Device BVHNodeInfo*     treeInternalNodes,
  const Device uint*            leafParentNodeIndices,
  const Device uint*            nodeParentNodeIndices,
  const Device XAB*             treeInternalNodeBoundingBoxes,
  const float3                  rayOrigin,
  const float3                  rayDirection,
  const float3                  invRayDirection,
  const bool3                   sign,
  const Device PrimitiveStruct* vertexArray,
  const Device PrimitiveAttrib* attributeArray,
  Const RTSystemSettings*       systemSettings)
{
  HitStruct hit;
  initializeHit(&hit);

  hit.distance = currentTime;

  short stackTop = 0;
  uint traversalStack[64];

  // mark index of the root node internal
  uint currNodeIndex = setBVHInternalNodeMarker(false, 0);

  while (true)
  {
#ifdef BOUNDING_VOLUME_HIERARCHY_ADS_DEBUG_TREE_TRAVERSAL
    printf("Node: %d %d %d\n", index, currNodeIndex, isBVHLeafNode(currNodeIndex));
#endif

    // traverse while a leaf node is found
    while (!isBVHLeafNode(currNodeIndex))
    {
      const BVHNodeInfo node     = treeInternalNodes[removeBVHInternalNodeMarker(currNodeIndex)];
      //const XAB leftBoundingBox  = treeInternalNodeBoundingBoxes[select(removeBVHInternalNodeMarker(node.childLeft),  node.childLeft,  isBVHLeafNode(node.childLeft))];
      //const XAB rightBoundingBox = treeInternalNodeBoundingBoxes[select(removeBVHInternalNodeMarker(node.childRight), node.childRight, isBVHLeafNode(node.childRight))];
      const XAB leftBoundingBox  = treeInternalNodeBoundingBoxes[removeBVHInternalNodeMarker(node.childLeft)];
      const XAB rightBoundingBox = treeInternalNodeBoundingBoxes[removeBVHInternalNodeMarker(node.childRight)];

      if (isBVHLeafNode(node.childLeft) || rayXABIntersectTest(hit.distance, leftBoundingBox, rayOrigin, invRayDirection, sign))
      //if (rayXABIntersectTest(hit.distance, leftBoundingBox, rayOrigin, invRayDirection, sign))
      {
        traversalStack[stackTop++] = node.childLeft;
#ifdef BOUNDING_VOLUME_HIERARCHY_ADS_DEBUG_TREE_TRAVERSAL
        printf("Stack Push: %d %d\n", index, node.child[0]);
#endif
      }

      if (isBVHLeafNode(node.childRight) || rayXABIntersectTest(hit.distance, rightBoundingBox, rayOrigin, invRayDirection, sign))
      //if (rayXABIntersectTest(hit.distance, rightBoundingBox, rayOrigin, invRayDirection, sign))
      {
        traversalStack[stackTop++] = node.childRight;
#ifdef BOUNDING_VOLUME_HIERARCHY_ADS_DEBUG_TREE_TRAVERSAL
        printf("Stack Push: %d %d\n", index, node.child[1]);
#endif
      }

      // break if the stack is empty
      if (stackTop == 0)
      {
        // mark node invalid
        currNodeIndex = BOUNDING_VOLUME_HIERARCHY_ADS_ROOT_NODE_MARKER;
        break;
      }

      currNodeIndex = traversalStack[--stackTop];
#ifdef BOUNDING_VOLUME_HIERARCHY_ADS_DEBUG_TREE_TRAVERSAL
      printf("Stack Pop: %d %d\n", index, currNodeIndex);
#endif
    }

#ifdef BOUNDING_VOLUME_HIERARCHY_ADS_DEBUG_TREE_TRAVERSAL
    printf("Test: %d %d\n", index, currNodeIndex);
#endif

    // test colision if not an invalid node
    if (currNodeIndex != BOUNDING_VOLUME_HIERARCHY_ADS_ROOT_NODE_MARKER)
    {
      if (earliestIntersection(&hit, currNodeIndex, rayOrigin, rayDirection, invRayDirection, sign, vertexArray, attributeArray, systemSettings))
      {
#ifdef IntersectionTypeAny
        break;
#endif
      }
    }

    // exit if nothing to fetch
    if (stackTop == 0)
    {
      break;
    }

    // pop from the stack
    currNodeIndex = traversalStack[--stackTop];
  }

  return hit;
}

Kernel void intersectRaysBVH(
  Device HitStruct*             hits,
  const Device RayStruct*       rays,
  constantKernelInput(uint,     rayCount),
  const Device PrimitiveStruct* vertexArray,
  const Device PrimitiveAttrib* attributeArray,
  const Device BVHNodeInfo*     treeInternalNodes,
  const Device uint*            leafParentNodeIndices,
  const Device uint*            nodeParentNodeIndices,
  const Device XAB*             treeInternalNodeBoundingBoxes,
  Const RTSystemSettings*       systemSettings,
  constantKernelInput(uint,     primitiveCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index >= rayCount)
    return;

  const float3 rayOrigin = rays[index].origin;
  const float3 rayDirection = rays[index].direction;
  const float3 invRayDirection = 1.f / rayDirection;
  const bool3 sign = selectInput3(invRayDirection < 0.f);

  HitStruct hit;

  // within valid grid cell bounds
#ifdef BOUNDING_VOLUME_HIERARCHY_ADS_DEBUG_TREE_TRAVERSAL
  if (index == 0)
#endif
  {
    hit = stackTraverseBinaryTree(rays[index].maxDistance, treeInternalNodes, leafParentNodeIndices, nodeParentNodeIndices, treeInternalNodeBoundingBoxes,
      rayOrigin, rayDirection, invRayDirection, sign, vertexArray, attributeArray, systemSettings);
  }

#ifdef IntersectionTypeClosest
#ifdef HitStructNormal
  if (hit.primitiveIndex != -1)
  {
    hit.normal = normalize(hit.normal);
  }
#endif
  hits[index] = hit;
#endif
#ifdef IntersectionTypeAny
  hits[index].primitiveIndex = hit.primitiveIndex;
#endif
}

#endif
