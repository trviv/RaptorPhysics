#ifndef COLLISION_SOLVER_SHADER
#define COLLISION_SOLVER_SHADER

inline void atomicAddFloat3(Device float3 *destination, const float3 value)
{
  Device uint *uintDestination = (Device uint*)destination;

  for (short i=0; i<3; i++, uintDestination++)
  {
    uint existingValue = atomicLoad(uintDestination);
    float desiredValue = ((const Thread float*)&value)[i] + asFloat(existingValue);
    while (!atomicCmpXchg(uintDestination, existingValue, desiredValue))
    {
      desiredValue = ((const Thread float*)&value)[i] + asFloat(existingValue);
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
    float desiredValue = ((const Thread float*)&value)[i] + asFloat(existingValue);
    while (!atomicCmpXchgShared(uintDestination, existingValue, desiredValue))
    {
      desiredValue = ((const Thread float*)&value)[i] + asFloat(existingValue);
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
  const ushort                        stablizationPass,
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
      (*collisionCount)++;
    }

    if (particle->position.x < min.x)
    {
      ret.x = min.x - particle->position.x;
      (*collisionCount)++;
    }

    if (particle->position.x > max.x)
    {
      ret.x = max.x - particle->position.x;
      (*collisionCount)++;
    }

    if (particle->position.z < min.z)
    {
      ret.z = min.z - particle->position.z;
      (*collisionCount)++;
    }

    if (particle->position.z > max.z)
    {
      ret.z = max.z - particle->position.z;
      (*collisionCount)++;
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
inline bool shouldCheckForCollision(const short solverType, const uint selfParticleIndex, const uint otherParticleIndex, const Thread ParticleStruct* selfParticle, const Thread ParticleStruct* otherParticle, const bool differentCell)
{
#if defined(GRID_COLLISION_SOLVE_PAIR_ONCE) && !defined(GRID_COLLISION_SOLVER_SCATTER_PARTICLES)
  return ((solverType == SOLVER_FLUID) || (solverType == SOLVER_CLOTH) || (otherParticle->identity.identity != selfParticle->identity.identity)) && (differentCell || (otherParticleIndex < selfParticleIndex));
#else
  return ((solverType == SOLVER_FLUID) || (solverType == SOLVER_CLOTH) || (otherParticle->identity.identity != selfParticle->identity.identity)) && (otherParticleIndex != selfParticleIndex);
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
  const ushort                        stablizationPass,
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

    XAB particleBoundingBox;
    particleBoundingBox.min = particle.position;
    particleBoundingBox.max = particle.position;

#ifdef COLLISION_SOLVER_SET_PARTICLE_BOUNDING_BOXES
    particleBoundingBoxes[index] = particleBoundingBox;
#endif

    mergeXAB(&accumulatedBoundingBox, &particleBoundingBox);
  }

  particleGroupBoundingBoxes[threadIndex()] = accumulatedBoundingBox;
}

/*
@kernel Compute and store bounding boxes for each particle only based on collision data.
@param particleBoundingBoxes Particle bounding box array.
@param particles Particle positions.
@param particleCollisionData Array containing particle SDF mass and radius data.
@param nodeCount Total nodes in the solver.
*/
Kernel void createBoundingBoxesCollision(
  Device XAB*                         particleGroupBoundingBoxes,
  const Device ParticleStruct*        particles,
  const Device ParticleCollisionData* particleCollisionData,
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
    const ParticleStruct particle = particles[index];
    const float radius = particleCollisionData[index].radius;

    XAB particleBoundingBox;
    particleBoundingBox.min = particle.position - constructFloat3(radius);
    particleBoundingBox.max = particle.position + constructFloat3(radius);

    mergeXAB(&accumulatedBoundingBox, &particleBoundingBox);
  }

  particleGroupBoundingBoxes[threadIndex()] = accumulatedBoundingBox;
}

#endif
