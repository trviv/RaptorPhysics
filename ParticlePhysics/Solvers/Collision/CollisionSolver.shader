#ifndef COLLISION_SOLVER_SHADER
#define COLLISION_SOLVER_SHADER

inline uint arrange32Bits(uint x)
{
  //........ ........ ......12 3456789A  //x
  //....1..2 ..3..4.. 5..6..7. .8..9..A  //x after interleaving bits

  //......12 3456789A ......12 3456789A  //x ^ (x << 16)
  //11111111 ........ ........ 11111111  //0x FF 00 00 FF
  //......12 ........ ........ 3456789A  //x = (x ^ (x << 16)) & 0xFF0000FF;

  //......12 ........ 3456789A 3456789A  //x ^ (x <<  8)
  //......11 ........ 1111.... ....1111  //0x 03 00 F0 0F
  //......12 ........ 3456.... ....789A  //x = (x ^ (x <<  8)) & 0x0300F00F;

  //..12..12 ....3456 3456.... 789A789A  //x ^ (x <<  4)
  //......11 ....11.. ..11.... 11....11  //0x 03 0C 30 C3
  //......12 ....34.. ..56.... 78....9A  //x = (x ^ (x <<  4)) & 0x030C30C3;

  //....1212 ..3434.. 5656..78 78..9A9A  //x ^ (x <<  2)
  //....1..1 ..1..1.. 1..1..1. .1..1..1  //0x 09 24 92 49
  //....1..2 ..3..4.. 5..6..7. .8..9..A  //x = (x ^ (x <<  2)) & 0x09249249;

  //........ ........ ......11 11111111  //0x000003FF

  x = (x ^ (x << 16)) & 0xFF0000FF;
  x = (x ^ (x << 8)) & 0x0300F00F;
  x = (x ^ (x << 4)) & 0x030C30C3;
  x = (x ^ (x << 2)) & 0x09249249;

  return x;
}

#define COLLISION_COMPONENT_MORTON_CODE_MASK 1023

inline uint get32BitMortonCode(const int3 quantizedPosition)
{
  const uint x = quantizedPosition.x & COLLISION_COMPONENT_MORTON_CODE_MASK;
  const uint y = quantizedPosition.y & COLLISION_COMPONENT_MORTON_CODE_MASK;
  const uint z = quantizedPosition.z & COLLISION_COMPONENT_MORTON_CODE_MASK;

  return arrange32Bits(x) | (arrange32Bits(y) << 1) | (arrange32Bits(z) << 2);
}

//#define MARK_COLLIDED_PARTICLES

#define BOUNDARY_BOTTOM   0.f
#define BOUNDARY_LEFT     -20.f
#define BOUNDARY_RIGHT    20.f
#define BOUNDARY_FRONT    20.f
#define BOUNDARY_BACK     -20.f
//#define BOUNDARY_LEFT     -2.f
//#define BOUNDARY_RIGHT    2.f
//#define BOUNDARY_FRONT    -1.f
//#define BOUNDARY_BACK     -7.f

/*
@kernel Apply boundary constrain.
@param particles Initial particle buffer.
@param collisionData Particle SDF mass and radius data.
*/
inline float3 boundaryCollision(
  Thread ParticleStruct*              particle,
  const Thread ParticleCollisionData* collisionData)
{
  float3 ret = constructFloat3(0.f);

  if (collisionData->invMass)  // only if movable
  {
    if (particle->position.y <= BOUNDARY_BOTTOM)
    {
      ret.y = BOUNDARY_BOTTOM - particle->position.y;
    }

    if (particle->position.x <= BOUNDARY_LEFT)
    {
      ret.x = BOUNDARY_LEFT - particle->position.x;
    }

    if (particle->position.x >= BOUNDARY_RIGHT)
    {
      ret.x = BOUNDARY_RIGHT - particle->position.x;
    }

    if (particle->position.z >= BOUNDARY_FRONT)
    {
      ret.z = BOUNDARY_FRONT - particle->position.z;
    }

    if (particle->position.z <= BOUNDARY_BACK)
    {
      ret.z = BOUNDARY_BACK - particle->position.z;
    }
  }

  return ret;
}

// function to process particle collision
inline float3 processParticleCollision(
  const Thread ParticleStruct* currentParticle,
  const Thread ParticleStruct* particleInit,
  const Thread ParticleStruct* otherParticle,
  const Thread ParticleStruct* otherParticleInit,
  const Thread ParticleCollisionData* collisionData,
  const Thread ParticleSharedData* sharedData,
  const uint currentNodeIndex,
  const uint index,
  const float sdfMagnitude,
  Thread short* collisionCount,
#ifdef MARK_COLLIDED_PARTICLES
  Device ParticleCollisionData* particleCollisionData,
  Thread bool* collided)
#else
  const Device ParticleCollisionData* particleCollisionData)
#endif
{
  if (otherParticle->identity.identity != currentParticle->identity.identity)
//    || (getSolverType(otherParticle->identity) == SOLVER_FLUID && currentNodeIndex != index))
  {
    const ParticleCollisionData collisionData2 = particleCollisionData[currentNodeIndex];
    const float sdfMagnitude2 = length(collisionData2.transformedSdfGradient);

    // skip if the base and the batch particle are of the same object
    float3 collisionVector = currentParticle->position - otherParticle->position;
    float actualDistance = dot(collisionVector, collisionVector);

#ifdef MARK_COLLIDED_PARTICLES
    const float allowedDistance = sqr(fabs(collisionData2.radius) + fabs(collisionData->radius));
#else
    const float allowedDistance = sqr(collisionData2.radius + collisionData->radius);
#endif

    // if overlapping
    if (actualDistance < allowedDistance)
    {
      actualDistance = sqrt(actualDistance);
      collisionVector /= actualDistance;

      // displacement magnitude
      float separationDistance = actualDistance - (fabs(collisionData->radius) + fabs(collisionData2.radius));

      // get normal according to minimum translation distance
      float3 sdfGradient = select(-collisionData2.transformedSdfGradient, collisionData->transformedSdfGradient, selectInput3(sdfMagnitude < sdfMagnitude2));
      sdfGradient = normalize(sdfGradient);

      float3 contactNormal = collisionVector;

      // sample signed distance field and modify normal
//      const float collDot = dot(sdfGradient, collisionVector);
//      if (collDot < 0.f)
//      {
//        contactNormal = collisionVector - (2.f * collDot) * contactNormal;
////        separationDistance = actualDistance - (collisionData.radius + collisionData2.radius);
//      }
//      else
//      {
//        contactNormal = collisionVector;
//      }
//      contactNormal = normalize(contactNormal);

#ifdef MARK_COLLIDED_PARTICLES
      *collided = true;
#endif
      (*collisionCount)++;

      float3 displacement1 = -separationDistance * contactNormal * (collisionData->invMass / (collisionData->invMass + collisionData2.invMass));
      float3 displacement2 = separationDistance * contactNormal * (collisionData2.invMass / (collisionData->invMass + collisionData2.invMass));
      float3 tangent = (displacement1 + currentParticle->position - particleInit->position) - (displacement2 + otherParticle->position - otherParticleInit->position);
      tangent = tangent - dot(tangent, sdfGradient) * sdfGradient;

      float tangentLength = length(tangent);

      if (tangentLength > COMPUTE_EPSILON)
      {
        const float minSdf = select(sdfMagnitude2, sdfMagnitude, sdfMagnitude < sdfMagnitude2);
        float displacementScale = select(min(sharedData->kineticFrictionCoef * separationDistance/tangentLength, 1.f), 1.f, tangentLength < sharedData->staticFrictionCoef * minSdf);
//        displacement1 -= tangent * displacementScale * (collisionData->invMass / (collisionData->invMass + collisionData2.invMass));
      }

      return displacement1;
    }
  }

  return constructFloat3(0.f);
}

/*
@kernel Get radius for particles, accumulate and store to an array.
@param groupRadius Maximum radius from the threadgroup.
@param particles Integrated particle position.
@param particleSharedData Particle entity shared data.
@param particleAuxData Additional particle data.
@param partitions Instance partition data.
@param entityLocation Entity section data.
@param globalOffsets Offsets to particle nodes all the solvers.
@param nodeBatchCount Total number of node batches.
@param nodeCount Total nodes in the solver.
*/
Kernel void getSystemMaxRadius(
  Device float*                     groupRadius,
  const Device ParticleStruct*      particles,
  const Device ParticleSharedData*  particleSharedData,
  const Device ParticleAuxData*     particleAuxData,
  const Device PartitionInfo*       partitions,
  const Device EntityLocation*      entityLocation,
  Const PhySystemOffsets*           globalOffsets,
  constantKernelInput(uint,         nodeBatchCount),
  constantKernelInput(uint,         nodeCount)
  KERNEL_GLOBAL_ARGUMENTS
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  // radius for the batch
  float maxRadius = 0.f;

  for (uint index = threadIndex(); index < nodeCount; index += threadGroupCount() * threadGroupSize())
  {
    const ParticleStruct particle = particles[index];
    ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(particle.identity);
    const PhySystemOffsets phySystemOffsets = globalOffsets[nodeIdentity.solverType];

    nodeIdentity.entityId += phySystemOffsets.globalSolverOffset;
    nodeIdentity.instanceId += phySystemOffsets.globalInstanceOffset;

    const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];
    const ParticleNodeLocator nodeLocator = getNodeLocator(index, phySystemOffsets.globalNodeOffset + partitions[nodeIdentity.instanceId].offset, entityLocation[nodeIdentity.entityId].node);

    maxRadius = max(maxRadius, getRadiusUsingDeviceAux(&sharedData, particleAuxData, nodeLocator.commonNodeIndex));
  }

  if (threadIndex() < nodeBatchCount)
  {
    groupRadius[threadIndex()] = maxRadius;
  }
}

/*
@kernel Compute and store bounding boxes for each particle.
@param particleBoundingBoxes Particle bounding box array.
@param particlesPredicted Integrated particle position.
@param particleSharedData Particle entity shared data.
@param particleAuxData Additional particle data.
@param partitions Instance partition data.
@param entityLocation Entity section data.
@param globalOffsets Offsets to particle nodes all the solvers.
@param nodeCount Total nodes in the solver.
*/
Kernel void createBoundingBoxes(
#ifdef COLLISION_SOLVER_SET_PARTICLE_BOUNDING_BOXES
  Device XAB*                       particleBoundingBoxes,
#endif
  Device XAB*                       particleGroupBoundingBoxes,
  const Device ParticleStruct*      particlesPredicted,
  const Device ParticleSharedData*  particleSharedData,
  const Device ParticleAuxData*     particleAuxData,
  const Device PartitionInfo*       partitions,
  const Device EntityLocation*      entityLocation,
#ifdef COLLISION_SOLVER_USE_SYSTEM_OFFSETS
  Const PhySystemOffsets*           globalOffsets,
#endif
  constantKernelInput(uint,         nodeBatchCount),
  constantKernelInput(uint,         nodeCount)
  KERNEL_GLOBAL_ARGUMENTS
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  // bounding box for the batch
  XAB accumulatedBoundingBox;

  // reset to INF, -INF
  clearXAB(&accumulatedBoundingBox, INFINITY);

  for (uint index = threadIndex(); index < nodeCount; index += threadGroupCount() * threadGroupSize())
  {
    const ParticleStruct particle = particlesPredicted[index];
    ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(particle.identity);
#ifdef COLLISION_SOLVER_USE_SYSTEM_OFFSETS
    const PhySystemOffsets phySystemOffsets = globalOffsets[nodeIdentity.solverType];

    nodeIdentity.entityId += phySystemOffsets.globalSolverOffset;
    nodeIdentity.instanceId += phySystemOffsets.globalInstanceOffset;
#endif

    const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];
#ifdef COLLISION_SOLVER_USE_SYSTEM_OFFSETS
    const ParticleNodeLocator nodeLocator = getNodeLocator(index, phySystemOffsets.globalNodeOffset + partitions[nodeIdentity.instanceId].offset, entityLocation[nodeIdentity.entityId].node);
#else
    const ParticleNodeLocator nodeLocator = getNodeLocator(index, partitions[nodeIdentity.instanceId].offset, entityLocation[nodeIdentity.entityId].node);
#endif

    const float radius = getRadiusUsingDeviceAux(&sharedData, particleAuxData, nodeLocator.commonNodeIndex);

    XAB particleBoundingBox;
    particleBoundingBox.min = particle.position - constructFloat3(radius);
    particleBoundingBox.max = particle.position + constructFloat3(radius);

#ifdef COLLISION_SOLVER_SET_PARTICLE_BOUNDING_BOXES
    particleBoundingBoxes[index] = particleBoundingBox;
#endif

    mergeXAB(&accumulatedBoundingBox, &particleBoundingBox);
  }

  if (threadIndex() < nodeBatchCount)
  {
    particleGroupBoundingBoxes[threadIndex()] = accumulatedBoundingBox;
  }
}

#endif
