#ifndef FLUID_SOLVER_PCISPH_SHADER
#define FLUID_SOLVER_PCISPH_SHADER

inline float poly6Function(const float r, const float h)
{
  const float x = (h * h - r * r) / (h * h * h);
  return (315.f / (64.f * M_PI_F)) * x * x * x;
}

inline float spikyFunction(const float r, const float h)
{
  const float x = (h - r) / (h * h);
  return (15.f / M_PI_F) * x * x * x;
}

inline float spikyFunctionGradient(const float r, const float h)
{
  const float x = (h - r) / (h * h * h);
  return -(45.f / M_PI_F) * x * x;
}

/*
@kernel Predict position and velocity for the next step.
@param particlesNextPosition Next particle positions.
@param particlesNextVelocity Next particle velocities.
@param particlesPosition Integrated particle position.
@param nodeCount Total nodes in the solver.
*/
Kernel void predictionStep(
  Device ParticleStruct*              particlesNextPosition,
  Device ParticleDifferential*        particlesNextVelocity,
  const Device ParticleStruct*        particlesPosition,
  const Device ParticleDifferential*  particlesDiff,
  const Device float4*                particlesPressureForce,
  const Device ParticleSharedData*    particleSharedData,
  constantKernelInput(uint,           nodeCount),
  constantKernelInput(float,          timeStep)
  KERNEL_GLOBAL_ARGUMENTS
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  const uint particleIndex = threadIndex();

  if (particleIndex >= nodeCount)
  {
    return;
  }

  // current particle data
  ParticleStruct selfParticle = particlesPosition[particleIndex];
  const IdentityInfo identity = selfParticle.identity;
  const ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(identity);
  const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];

  float3 velocity = particlesDiff[particleIndex].velocity + particlesPressureForce[particleIndex].xyz * timeStep * sharedData.sharedInvMass;
  particlesNextVelocity[particleIndex].velocity = velocity;

  selfParticle.position += velocity * timeStep;
  selfParticle.identity = identity;
  particlesNextPosition[particleIndex] = selfParticle;
}

/*
@kernel Get particle densities.
@param particlesDensity Particle densities output.
@param gridCellParticleOffsets Starting offset for each grid cell.
@param gridCellParticleIndices Output array for particle indices.
@param gridParticleCellIndex Computed cell index for each particle.
@param particlesPredictedOld Integrated particle position.
@param particleSharedData Particle entity shared data.
@param systemBoundingBox Physics system's bounding box.
@param invRadius Inverse of max particle radius in the system.
@param gridSize Size of grid in one dimension.
@param gridSizeExp Grid size in power of 2.
@param nodeCount Total nodes in the solver.
*/
Kernel void calculateDensity(
  Device float*                     particlesDensity,
  const Device uint*                gridCellParticleOffsets,
  const Device uint*                gridCellParticleIndices,
  const Device uint*                gridParticleCellIndex,
  const Device ParticleStruct*      particlesPredicted,
  const Device ParticleSharedData*  particleSharedData,
  Const XAB*                        systemBoundingBox,
  Const float*                      invRadius,
  constantKernelInput(int,          gridSize),
  constantKernelInput(int,          gridSizeExp),
  constantKernelInput(uint,         nodeCount),
  constantKernelInput(float,        timeStep)
  KERNEL_GLOBAL_ARGUMENTS
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  uint particleIndex = threadIndex();

  if (particleIndex >= nodeCount)
  {
    return;
  }

  particleIndex = gridCellParticleIndices[particleIndex];

  const uint gridCellIndex = gridParticleCellIndex[particleIndex];

  // current particle data
  float density = 0.f;

  const ParticleStruct selfParticle = particlesPredicted[particleIndex];
  const IdentityInfo identity = selfParticle.identity;
  const ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(identity);
  const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];

  const short3 particleGridCellIndex = constructShort3(
    gridCellIndex & (gridSize - 1),
    (gridCellIndex >> gridSizeExp) & (gridSize - 1),
    gridCellIndex >> (gridSizeExp << 1)
  );

#ifdef GRID_SOLVER_HASH_FUNCTION
  const float3 particleCellPosition = (selfParticle.position - systemBoundingBox->min) * invRadius[0];
#endif

  GRID_SOLVER_NEIGHBOUR_LOOP_BEGIN
    const uint2 indexRange = getRangeFromOffset(gridCellParticleOffsets, gridCellIndex);

    // batchwise iterate over indices in the cell
    for (int otherParticlePointerIndex = indexRange.x; otherParticlePointerIndex < indexRange.y; otherParticlePointerIndex++)
    {
      // iterate over each particle in the loaded batch
      const int otherNodeIndex = gridCellParticleIndices[otherParticlePointerIndex];
      const ParticleStruct otherParticle = particlesPredicted[otherNodeIndex];

      const float3 collisionVector = selfParticle.position - otherParticle.position;
      float actualDistance = length(collisionVector);

      density += select(0.f, poly6Function(actualDistance, sharedData.fluidKernelRadius), actualDistance < sharedData.fluidKernelRadius);
    }
  GRID_SOLVER_NEIGHBOUR_LOOP_END

  particlesDensity[particleIndex] = density / sharedData.sharedInvMass;
}

/*
@kernel Get particle pressures.
@param particlesPressure Particle pressures output.
@param gridCellParticleOffsets Starting offset for each grid cell.
@param gridCellParticleIndices Output array for particle indices.
@param gridParticleCellIndex Computed cell index for each particle.
@param particlesPredictedOld Integrated particle position.
@param particleSharedData Particle entity shared data.
@param systemBoundingBox Physics system's bounding box.
@param invRadius Inverse of max particle radius in the system.
@param gridSize Size of grid in one dimension.
@param gridSizeExp Grid size in power of 2.
@param nodeCount Total nodes in the solver.
*/
Kernel void calculatePressure(
  Device float*                     particlesPressure,
  const Device uint*                gridCellParticleOffsets,
  const Device uint*                gridCellParticleIndices,
  const Device uint*                gridParticleCellIndex,
  const Device ParticleStruct*      particlesPredicted,
  const Device ParticleSharedData*  particleSharedData,
  Const XAB*                        systemBoundingBox,
  Const float*                      invRadius,
  constantKernelInput(int,          gridSize),
  constantKernelInput(int,          gridSizeExp),
  constantKernelInput(uint,         nodeCount),
  constantKernelInput(float,        timeStep),
  constantKernelInput(float,        beta)
  KERNEL_GLOBAL_ARGUMENTS
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{

  uint particleIndex = threadIndex();

  if (particleIndex >= nodeCount)
  {
    return;
  }

  particleIndex = gridCellParticleIndices[particleIndex];

  const uint gridCellIndex = gridParticleCellIndex[particleIndex];

  // current particle data
  float density = 0.f;
  float sumGradientMagnitude = 0.f;
  float3 sumGradientVector = constructFloat3(0.f);

  const ParticleStruct selfParticle = particlesPredicted[particleIndex];
  const IdentityInfo identity = selfParticle.identity;
  const ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(identity);
  const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];

  const short3 particleGridCellIndex = constructShort3(
    gridCellIndex & (gridSize - 1),
    (gridCellIndex >> gridSizeExp) & (gridSize - 1),
    gridCellIndex >> (gridSizeExp << 1)
  );

#ifdef GRID_SOLVER_HASH_FUNCTION
  const float3 particleCellPosition = (selfParticle.position - systemBoundingBox->min) * invRadius[0];
#endif

  GRID_SOLVER_NEIGHBOUR_LOOP_BEGIN
    const uint2 indexRange = getRangeFromOffset(gridCellParticleOffsets, gridCellIndex);

    // batchwise iterate over indices in the cell
    for (int otherParticlePointerIndex = indexRange.x; otherParticlePointerIndex < indexRange.y; otherParticlePointerIndex++)
    {
      // iterate over each particle in the loaded batch
      const int otherNodeIndex = gridCellParticleIndices[otherParticlePointerIndex];
      const ParticleStruct otherParticle = particlesPredicted[otherNodeIndex];

      const float3 collisionVector = selfParticle.position - otherParticle.position;
      float actualDistance = length(collisionVector);

      density += select(0.f, poly6Function(actualDistance, sharedData.fluidKernelRadius), actualDistance < sharedData.fluidKernelRadius);

      if (actualDistance >= sharedData.fluidKernelRadius || actualDistance <= COMPUTE_EPSILON || otherNodeIndex == particleIndex)
        continue;

      const float3 gradient = collisionVector * spikyFunctionGradient(actualDistance, sharedData.fluidKernelRadius);
      sumGradientMagnitude += lengthSq(gradient);
      sumGradientVector += gradient;
    }
  GRID_SOLVER_NEIGHBOUR_LOOP_END

  density /= sharedData.sharedInvMass;

  const float deltaDenom = (beta * (-dot(sumGradientVector, sumGradientVector) - sumGradientMagnitude));

  if (fabs(deltaDenom) > COMPUTE_EPSILON * COMPUTE_EPSILON)
  {
    particlesPressure[particleIndex] += -(density - 1.f/sharedData.invRestDensity) / deltaDenom;
  }
}

/*
@kernel Calculate particle forces.
@param particlesPredictedNew Updated particle positions post collision processing.
@param particlesDensity Particle densities output.
@param particlesPressure Particle pressures output.
@param gridCellParticleOffsets Starting offset for each grid cell.
@param gridCellParticleIndices Output array for particle indices.
@param particlesPosition Integrated particle position.
@param particleSharedData Particle entity shared data.
@param partitions Instance partition data.
@param entityLocation Entity section data.
@param gridParticleCellIndex Computed cell index for each particle.
@param nodeCount Total nodes in the solver.
@param occupiedCellCount Total active grid cells.
*/
Kernel void calculateForces(
  Device float4*                      particlesPressureForce,
  const Device float*                 particlesDensity,
  const Device float*                 particlesPressure,
  const Device uint*                  gridCellParticleOffsets,
  const Device uint*                  gridCellParticleIndices,
  const Device uint*                  gridParticleCellIndex,
  const Device ParticleStruct*        particlesPosition,
  const Device ParticleSharedData*    particleSharedData,
  Const XAB*                          systemBoundingBox,
  Const float*                        invRadius,
  constantKernelInput(int,            gridSize),
  constantKernelInput(int,            gridSizeExp),
  constantKernelInput(uint,           nodeCount),
  constantKernelInput(float,          timeStep)
  KERNEL_GLOBAL_ARGUMENTS
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  uint particleIndex = threadIndex();

  if (particleIndex >= nodeCount)
  {
    return;
  }

  particleIndex = gridCellParticleIndices[particleIndex];

  const uint gridCellIndex = gridParticleCellIndex[particleIndex];

  float3 force = constructFloat3(0.f);

  // current particle data
  ParticleStruct selfParticle = particlesPosition[particleIndex];
  const IdentityInfo identity = selfParticle.identity;
  const ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(identity);
  const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];

  const float selfDensity = particlesDensity[particleIndex];
  const float selfPressure = particlesPressure[particleIndex];

  const short3 particleGridCellIndex = constructShort3(
    gridCellIndex & (gridSize - 1),
    (gridCellIndex >> gridSizeExp) & (gridSize - 1),
    gridCellIndex >> (gridSizeExp << 1)
  );

#ifdef GRID_SOLVER_HASH_FUNCTION
  const float3 particleCellPosition = (selfParticle.position - systemBoundingBox->min) * invRadius[0];
#endif

  const float selfPressureByDensity = selfPressure/sqr(selfDensity);

  GRID_SOLVER_NEIGHBOUR_LOOP_BEGIN
    const uint2 indexRange = getRangeFromOffset(gridCellParticleOffsets, gridCellIndex);

    // batchwise iterate over indices in the cell
    for (int otherParticlePointerIndex = indexRange.x; otherParticlePointerIndex < indexRange.y; otherParticlePointerIndex++)
    {
      // iterate over each particle in the loaded batch
      const int otherNodeIndex = gridCellParticleIndices[otherParticlePointerIndex];
      const ParticleStruct otherParticle = particlesPosition[otherNodeIndex];

      const float3 collisionVector = selfParticle.position - otherParticle.position;
      float actualDistance = length(collisionVector);

      if (actualDistance >= sharedData.fluidKernelRadius || actualDistance <= COMPUTE_EPSILON || otherNodeIndex == particleIndex)
      {
        continue;
      }

      const float density = selfPressureByDensity + particlesPressure[otherNodeIndex]/sqr(particlesDensity[otherNodeIndex]);
      force += collisionVector * (density * spikyFunctionGradient(actualDistance, sharedData.fluidKernelRadius));
    }
  GRID_SOLVER_NEIGHBOUR_LOOP_END

  particlesPressureForce[particleIndex].xyz = -force / sqr(sharedData.sharedInvMass);
}

/*
@kernel Update positions based on pressure force.
@param particlesPosition Particle positions.
@param particlesPressureForce Particle forces calculated from pressure.
@param particleSharedData Particle entity shared data.
@param partitions Instance partition data.
@param gridParticleCellIndex Computed cell index for each particle.
@param nodeCount Total nodes in the solver.
@param occupiedCellCount Total active grid cells.
*/
Kernel void updatePositions(
  Device ParticleStruct*              particlesPosition,
  const Device float4*                particlesPressureForce,
  const Device ParticleSharedData*    particleSharedData,
  constantKernelInput(uint,           nodeCount),
  constantKernelInput(float,          timeStep)
  KERNEL_GLOBAL_ARGUMENTS
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  const uint particleIndex = threadIndex();

  if (particleIndex >= nodeCount)
  {
    return;
  }

  const float3 force = particlesPressureForce[particleIndex].xyz;

  // current particle data
  ParticleStruct selfParticle = particlesPosition[particleIndex];
  const IdentityInfo identity = selfParticle.identity;
  const ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(identity);
  const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];

  selfParticle.position += force * sqr(timeStep) * sharedData.sharedInvMass;
  selfParticle.identity = identity;
  particlesPosition[particleIndex] = selfParticle;
}
#endif
