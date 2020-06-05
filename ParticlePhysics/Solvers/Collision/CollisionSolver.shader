#ifndef COLLISION_SOLVER_SHADER
#define COLLISION_SOLVER_SHADER

inline uint encode32Bits(uint x)
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

  return encode32Bits(x) | (encode32Bits(y) << 1) | (encode32Bits(z) << 2);
}

inline uint decode32Bits(uint x)
{
  x &= 0x09249249;                  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
  x = (x ^ (x >>  2)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
  x = (x ^ (x >>  4)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
  x = (x ^ (x >>  8)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
  x = (x ^ (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210

  return x;
}

inline int3 decode32BitMortonCode(const uint mortonCode)
{
  return constructInt3(decode32Bits(mortonCode), decode32Bits(mortonCode >> 1), decode32Bits(mortonCode >> 2));
}

inline void atomicAddFloat3(Device float3 *destination, const float3 value)
{
  Device uint *uintDestination = (Device uint*)destination;

  for (short i=0; i<3; i++, uintDestination++)
  {
    uint existingValue = atomicLoad(uintDestination);
    float desiredValue = value[i] + asFloat(existingValue);
    while (!atomicCmpXchg(uintDestination, existingValue, desiredValue))
    {
      desiredValue = value[i] + asFloat(existingValue);
    }
  }
}

inline float3 atomicLoadFloat3(const Device float3 *source)
{
  const Device uint *uintSource = (const Device uint*)source;

  return constructFloat3(asFloat(atomicLoad(&uintSource[0])), asFloat(atomicLoad(&uintSource[1])), asFloat(atomicLoad(&uintSource[2])));
}

inline void atomicAddFloat3Shared(Shared float3 *destination, const float3 value)
{
  Shared uint *uintDestination = (Shared uint*)destination;

  for (short i=0; i<3; i++, uintDestination++)
  {
    uint existingValue = atomicLoadShared(uintDestination);
    float desiredValue = value[i] + asFloat(existingValue);
    while (!atomicCmpXchgShared(uintDestination, existingValue, desiredValue))
    {
      desiredValue = value[i] + asFloat(existingValue);
    }
  }
}

inline void atomicAddFloat(Device float *destination, const float value)
{
  Device uint *uintDestination = (Device uint*)destination;

  uint existingValue = atomicLoad(uintDestination);
  float desiredValue = value + asFloat(existingValue);
  while (!atomicCmpXchg(uintDestination, existingValue, desiredValue))
  {
    desiredValue = value + asFloat(existingValue);
  }
}

inline void atomicAddFloatShared(Shared float *destination, const float value)
{
  Shared uint *uintDestination = (Shared uint*)destination;

  uint existingValue = atomicLoadShared(uintDestination);
  float desiredValue = value + asFloat(existingValue);
  while (!atomicCmpXchgShared(uintDestination, existingValue, desiredValue))
  {
    desiredValue = value + asFloat(existingValue);
  }
}

//#define MARK_COLLIDED_PARTICLES

#ifdef GRID_COLLISION_SOLVER_USE_SHARED_MEMORY
#define Scope Shared
#else
#define Scope Thread
#endif

inline float3 calculateFriction(
  const float3 selfParticleVelocity,
  const float3 otherParticleVelocity,
  const float3 contactNormal,
  const float separationDistance,
  const Thread CollisionSolverData* collisionSolverData)
{
  float3 tangent = selfParticleVelocity - otherParticleVelocity;
  tangent = tangent - dot(tangent, contactNormal) * contactNormal;

  const float tangentLength = length(tangent);
  const float staticFactor = collisionSolverData->staticFrictionCoef * separationDistance;
  const float kineticFactor = collisionSolverData->kineticFrictionCoef * separationDistance;

  return tangent * select(0.f, select(min(kineticFactor / tangentLength, 1.f), 1.f, tangentLength < staticFactor), tangentLength > COMPUTE_EPSILON);
}

/*
@kernel Apply boundary constrain.
@param particles Initial particle buffer.
@param collisionData Particle SDF mass and radius data.
*/
inline float3 boundaryCollision(
  Thread ParticleStruct*              particle,
  const Thread ParticleDifferential*  selfParticleDiff,
  const Thread ParticleCollisionData* collisionData,
  Const PhySystemSettings*            systemSettings,
  const uint                          stablizationPass,
  Thread uint*                        collisionCount,
#ifdef MARK_COLLIDED_PARTICLES
  Device ParticleCollisionData*       particleCollisionData,
#else
  const Device ParticleCollisionData* particleCollisionData,
#endif
  const Thread CollisionSolverData*   collisionSolverData)
{
  float3 ret = constructFloat3(0.f);
#ifdef MARK_COLLIDED_PARTICLES
  const float3 min = systemSettings->systemBound.min + constructFloat3(fabs(collisionData->radius));
  const float3 max = systemSettings->systemBound.max - constructFloat3(fabs(collisionData->radius));
#else
  const float3 min = systemSettings->systemBound.min + constructFloat3(collisionData->radius);
  const float3 max = systemSettings->systemBound.max - constructFloat3(collisionData->radius);
#endif

  // only if movable
  if (collisionData->invMass)
  {
    if (particle->position.y < min.y)
    {
      ret.y = min.y - particle->position.y;
      *collisionCount++;
    }

    if (particle->position.x < min.x)
    {
      ret.x = min.x - particle->position.x;
      *collisionCount++;
    }

    if (particle->position.x > max.x)
    {
      ret.x = max.x - particle->position.x;
      *collisionCount++;
    }

    if (particle->position.z < min.z)
    {
      ret.z = min.z - particle->position.z;
      *collisionCount++;
    }

    if (particle->position.z > max.z)
    {
      ret.z = max.z - particle->position.z;
      *collisionCount++;
    }

    if (dot(ret, ret) > 0.f && SOLVER_FLUID != getSolverType(particle->identity))
    {
      const float3 friction = calculateFriction(selfParticleDiff->velocity, constructFloat3(0.f), -normalize(ret), length(ret), collisionSolverData);
      if (!stablizationPass)
      {
        ret -= friction;
      }

#ifdef MARK_COLLIDED_PARTICLES
      float invMass = particleCollisionData->invMass;
      particleCollisionData->transformedSdfGradient = encodeDirection(2.f * fabs(collisionData->radius) * normalize(friction));
      particleCollisionData->invMass = invMass;
#endif
    }
  }

  return ret;
}

// function to check if objects are eligible for collision
inline bool shouldCheckForCollision(const short solverType, const uint selfParticleIndex, const uint otherParticleIndex, const Thread ParticleStruct* selfParticle, const Thread ParticleStruct* otherParticle, const bool differentCell = true)
{
#if defined(GRID_COLLISION_SOLVE_PAIR_ONCE) && !defined(GRID_COLLISION_SOLVER_SCATTER_PARTICLES)
  return (differentCell | otherParticleIndex < selfParticleIndex) & (solverType == SOLVER_FLUID | solverType == SOLVER_CLOTH | otherParticle->identity.identity != selfParticle->identity.identity);
#else
  return otherParticleIndex != selfParticleIndex & (solverType == SOLVER_FLUID | solverType == SOLVER_CLOTH | otherParticle->identity.identity != selfParticle->identity.identity);
#endif
}

// function to process particle collision
inline float3 processParticleCollision(
  const Thread ParticleStruct*        selfParticle,
  const Thread ParticleDifferential*  selfParticleDiff,
  const Scope ParticleStruct*         otherParticle,
  const Thread ParticleDifferential*  otherParticleDiff,
  const bool                          updateOtherParticle,
  const Thread ParticleCollisionData* collisionData,
  const Thread CollisionSolverData*   collisionSolverData,
  const uint                          currentNodeIndex,
  const uint                          index,
  const float                         sdfMagnitude,
  Thread uint*                        collisionCount,
  const uint                          stablizationPass,
  const short                         solverType,
  Device ParticleStruct*              particlesDelta,
#ifdef MARK_COLLIDED_PARTICLES
  Device ParticleCollisionData*       particleCollisionData)
#else
  const Device ParticleCollisionData* particleCollisionData)
#endif
{
  const ParticleCollisionData collisionData2 = particleCollisionData[currentNodeIndex];
  const float sdfMagnitude2 = collisionData2.gradientMagnitude;

  // skip if the base and the batch particle are of the same object
  float3 collisionVector = selfParticle->position - otherParticle->position;
  float actualDistance = lengthSq(collisionVector);

#ifdef MARK_COLLIDED_PARTICLES
  const float allowedDistance = (fabs(collisionData2.radius) + fabs(collisionData->radius));
#else
  const float allowedDistance = (collisionData2.radius + collisionData->radius);
#endif

  // if overlapping
  if (actualDistance < (allowedDistance * allowedDistance))
  {
    actualDistance = sqrt(actualDistance);
    // TODO: Look into SDF
    // displacement magnitude
    const float separationDistance = actualDistance - allowedDistance + COMPUTE_EPSILON;

    // get normal according to minimum translation distance
    //float3 sdfGradient = select(-collisionData2.transformedSdfGradient, collisionData->transformedSdfGradient, selectInput3(sdfMagnitude < sdfMagnitude2));

    // sample signed distance field and modify normal
    //const float collDot = dot(sdfGradient, collisionVector);
    //float3 contactNormal = select(collisionVector, collisionVector - (2.f * collDot) * sdfGradient, selectInput3(collDot < 0.f));
    //contactNormal = normalize(contactNormal);

    float3 contactNormal = collisionVector / max(COMPUTE_EPSILON, actualDistance);

    (*collisionCount)++;

    const float massScale = 1.f / (collisionData->invMass + collisionData2.invMass);
    const float3 displacementFactor = contactNormal * separationDistance;

    float3 displacement = -displacementFactor;
    if (!stablizationPass && solverType != SOLVER_FLUID)
    {
      displacement += calculateFriction(selfParticleDiff->velocity, otherParticleDiff->velocity, contactNormal, separationDistance, collisionSolverData);
    }

    displacement *= massScale;
#ifdef MARK_COLLIDED_PARTICLES
    float invMass = particleCollisionData[index].invMass;
    particleCollisionData[index].transformedSdfGradient = encodeDirection(2.f * fabs(collisionData->radius) * normalize(contactNormal));
    particleCollisionData[index].invMass = invMass;
#endif

#ifdef GRID_COLLISION_SOLVE_PAIR_ONCE
    if (updateOtherParticle)
    {
      atomicAddFloat3(&particlesDelta[currentNodeIndex].position, -displacement * collisionData2.invMass * collisionSolverData->collisionDamping);
      atomicAdd(&particlesDelta[currentNodeIndex].identity.identity, 1);
    }
#endif

    return displacement * collisionData->invMass;
  }

  return constructFloat3(0.f);
}

/*
@kernel Get radius for particles, accumulate and store to an array.
@param groupRadius Maximum radius from the threadgroup.
@param particles Integrated particle position.
@param particleSharedData Particle entity shared data.
@param particleCollisionData Array containing particle SDF mass and radius data.
@param systemSettings Settings for the physics system.
@param nodeBatchCount Total number of node batches.
@param nodeCount Total nodes in the solver.
*/
Kernel void getSystemMaxRadius(
  Device float*                       groupRadius,
  const Device ParticleStruct*        particles,
  const Device ParticleSharedData*    particleSharedData,
  const Device ParticleCollisionData* particleCollisionData,
  Const PhySystemSettings*            systemSettings,
  constantKernelInput(uint,           nodeBatchCount),
  constantKernelInput(uint,           nodeCount)
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
    const PhySystemOffsets phySystemOffsets = systemSettings->globalOffsets[nodeIdentity.solverType];

    nodeIdentity.entityId += phySystemOffsets.globalSolverOffset;
    nodeIdentity.instanceId += phySystemOffsets.globalInstanceOffset;

    const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];

    maxRadius = max(maxRadius, getRadiusUsingDeviceCollision(&sharedData, particleCollisionData, index));
  }

  if (threadIndex() < nodeBatchCount)
  {
    groupRadius[threadIndex()] = 2.f * maxRadius;
  }
}

/*
@kernel Compute and store bounding boxes for each particle.
@param particleBoundingBoxes Particle bounding box array.
@param particlesPredicted Integrated particle position.
@param particleSharedData Particle entity shared data.
@param particleCollisionData Array containing particle SDF mass and radius data.
@param systemSettings Settings for the physics system.
@param nodeCount Total nodes in the solver.
*/
Kernel void createBoundingBoxes(
#ifdef COLLISION_SOLVER_SET_PARTICLE_BOUNDING_BOXES
  Device XAB*                         particleBoundingBoxes,
#endif
  Device XAB*                         particleGroupBoundingBoxes,
  const Device ParticleStruct*        particlesPredicted,
  const Device ParticleSharedData*    particleSharedData,
  const Device ParticleCollisionData* particleCollisionData,
#ifdef COLLISION_SOLVER_USE_SYSTEM_OFFSETS
  Const PhySystemSettings*            systemSettings,
#endif
  constantKernelInput(uint,           nodeBatchCount),
  constantKernelInput(uint,           nodeCount)
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
    const PhySystemOffsets phySystemOffsets = systemSettings->globalOffsets[nodeIdentity.solverType];

    nodeIdentity.entityId += phySystemOffsets.globalSolverOffset;
    nodeIdentity.instanceId += phySystemOffsets.globalInstanceOffset;
#endif

    const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];
    const float radius = getRadiusUsingDeviceCollision(&sharedData, particleCollisionData, index);

    XAB particleBoundingBox;
    particleBoundingBox.min = particle.position - constructFloat3(radius);
    particleBoundingBox.max = particle.position + constructFloat3(radius);

#ifdef COLLISION_SOLVER_SET_PARTICLE_BOUNDING_BOXES
    particleBoundingBoxes[index] = particleBoundingBox;
#endif

    mergeXAB(&accumulatedBoundingBox, &particleBoundingBox);
  }

  particleGroupBoundingBoxes[threadIndex()] = accumulatedBoundingBox;
}

#endif
