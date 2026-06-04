/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#ifndef FLUID_SOLVER_COMMON_SHADER
#define FLUID_SOLVER_COMMON_SHADER

#define FLUID_SOLVER_SORTED_REARRANGE
#define FLUID_SOLVER_SORTED_REARRANGE_MULTIPLIER 2

inline float poly6FunctionConstant(const float h)
{
  const float x = 1.f / (h * h * h);
  return (315.f / (64.f * M_PI_F)) * x * x * x;
}

inline float spikyFunctionConstant(const float h)
{
  const float x = 1.f / (h * h);
  return (15.f / M_PI_F) * x * x * x;
}

inline float viscosityFunctionConstant(const float h)
{
  const float x = 1.f / (h * h);
  return (45.f / M_PI_F) * x * x * x;
}

inline float poly6FunctionVariable(const float r, const float h)
{
  const float x = (h * h - r * r);
  return x * x * x;
}

inline float poly6FunctionVariableSquares(const float rSquared, const float hSquared)
{
  const float x = (hSquared - rSquared);
  return x * x * x;
}

inline float poly6FunctionGradientVariable(const float r, const float h)
{
  const float x = (h * h - r * r);
  return -6.f * x * x;
}

inline float poly6FunctionGradientVariableSquares(const float rSquared, const float hSquared)
{
  const float x = (hSquared - rSquared);
  return -6.f * x * x;
}

inline float poly6FunctionLaplacianVariable(const float r, const float h)
{
  const float x = (h * h - r * r) * (3 * h * h - 7 * r * r);
  return -6.f * x;
}

inline float poly6FunctionLaplacianVariableSquares(const float rSquared, const float hSquared)
{
  const float x = (hSquared - rSquared) * (3 * hSquared - 7 * rSquared);
  return -6.f * x;
}

inline float spikyFunctionVariable(const float r, const float h)
{
  const float x = (h - r);
  return x * x * x;
}

inline float spikyFunctionGradientVariable(const float r, const float h)
{
  const float x = (h - r);
  return -3.f * x * x;
}

inline float viscosityFunctionLaplacianVariable(const float r, const float h)
{
  const float x = (h - r);
  return x;
}

#ifdef COMPUTE_SHADER_SCOPE

inline void bitonicSortSharedUint3(
  Shared uint3* localNodes,
  const short   maxDepth,
  const short   localThreadCount,
  const short   localIndex)
{
  localMemBarrier();

  for (short mergeSize = 2; mergeSize <= (1 << maxDepth); mergeSize <<= 1)
  {
    for (short mergeSubSize = mergeSize>>1; mergeSubSize > 0; mergeSubSize >>= 1)
    {
      for (short m=0; m<FLUID_SOLVER_SORTED_REARRANGE_MULTIPLIER; m++)
      {
        const short indexLow  = (localIndex + localThreadCount * m) & (mergeSubSize - 1);
        const short indexHigh = (localIndex + localThreadCount * m - indexLow) << 1;
        const short index     = indexHigh + indexLow;
        const short swapIndex = indexHigh + select(mergeSubSize + indexLow, 2 * mergeSubSize - 1 - indexLow, mergeSubSize == (mergeSize >> 1));

        if (swapIndex < localThreadCount * FLUID_SOLVER_SORTED_REARRANGE_MULTIPLIER && index < localThreadCount * FLUID_SOLVER_SORTED_REARRANGE_MULTIPLIER)
        {
          if (localNodes[index].z == localNodes[swapIndex].z)
          {
            const uint2 node1 = localNodes[index].xy;
            const uint2 node2 = localNodes[swapIndex].xy;

            if (node1.x < node2.x)
            {
              localNodes[index].xy     = node2;
              localNodes[swapIndex].xy = node1;
            }
          }
        }
      }
      localMemBarrier();
    }
  }
}

/*
@kernel Kernel to reorder particles in buffers based on grid index.
@param gridCellParticleIndices Output array for particle indices.
@param nodeCount Total nodes in the solver.
*/
Kernel void reorderFluidParticles(
  Device ParticleStruct*              particlesNew,
  const Device ParticleStruct*        particlesOld,
  Device ParticleDifferential*        particleDiffNew,
  const Device ParticleDifferential*  particleDiffOld,
  Device ParticleStruct*              particlesPredictedNew,
  const Device ParticleStruct*        particlesPredictedOld,
  Device uint*                        gridParticleCellIndexNew,
  const Device uint*                  gridParticleCellIndexOld,
  const Device uint*                  gridCellParticleIndices,
#ifdef FLUID_SOLVER_SORTED_REARRANGE
  Const XAB*                          systemBoundingBox,
  Const float*                        invRadius,
  constantKernelInput(uint,           nodeCount),
  sharedMemKernelInput(uint3,         particleSpatialData,  12)
  KERNEL_GLOBAL_ARGUMENTS
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  const float scale = 32.f;

  for (short i=0; i<FLUID_SOLVER_SORTED_REARRANGE_MULTIPLIER; i++)
  {
    uint3 particleData = constructUint3(-1);
    const uint threadGlobalIndex = threadLocalIndex() + threadGroupSize() * (i + threadGroupIndex() * FLUID_SOLVER_SORTED_REARRANGE_MULTIPLIER);
    if (threadGlobalIndex < nodeCount)
    {
      const uint particleIndex = gridCellParticleIndices[threadGlobalIndex];
      const ParticleStruct selfParticle = particlesPredictedOld[particleIndex];
      const float3 _fracInput = (selfParticle.position - systemBoundingBox->min) * invRadius[0];
      const uint cellInternalSpatialIndex = encode16BitMortonCode(convertShort3((_fracInput - floor(_fracInput)) * scale));

      particleData = constructUint3(cellInternalSpatialIndex, particleIndex, gridParticleCellIndexOld[particleIndex]);
    }
    particleSpatialData[threadLocalIndex() + i * threadGroupSize()] = particleData;
  }

  const short tgSizePowOf2 = 32 - clz((int)threadGroupSize() * FLUID_SOLVER_SORTED_REARRANGE_MULTIPLIER) - 1;
  bitonicSortSharedUint3(particleSpatialData, tgSizePowOf2 / 2, threadGroupSize(), threadLocalIndex());

  for (short i=0; i<FLUID_SOLVER_SORTED_REARRANGE_MULTIPLIER; i++)
  {
    const uint particleIndex = particleSpatialData[threadLocalIndex() + i * threadGroupSize()].y;
    const uint threadGlobalIndex = threadLocalIndex() + threadGroupSize() * (i + threadGroupIndex() * FLUID_SOLVER_SORTED_REARRANGE_MULTIPLIER);

    if (particleIndex != -1)
    {
      gridParticleCellIndexNew[threadGlobalIndex] = particleSpatialData[threadLocalIndex() + i * threadGroupSize()].z;
      particlesNew[threadGlobalIndex]             = particlesOld[particleIndex];
      particlesPredictedNew[threadGlobalIndex]    = particlesPredictedOld[particleIndex];
      particleDiffNew[threadGlobalIndex]          = particleDiffOld[particleIndex];
    }
  }
#else
  constantKernelInput(uint,           nodeCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  uint particleIndex = threadIndex();

  if (particleIndex >= nodeCount)
  {
    return;
  }

  particleIndex = gridCellParticleIndices[threadIndex()];

  gridParticleCellIndexNew[threadIndex()] = gridParticleCellIndexOld[particleIndex];
  particlesNew[threadIndex()] = particlesOld[particleIndex];
  particlesPredictedNew[threadIndex()] = particlesPredictedOld[particleIndex];
  particleDiffNew[threadIndex()] = particleDiffOld[particleIndex];
#endif
}

/*
@kernel Kernel to reorder particles in buffers based on grid index.
@param gridCellParticleIndices Output array for particle indices.
@param nodeCount Total nodes in the solver.
*/
Kernel void reorderCouplingParticles(
  Device ParticleStruct*              particlesNew,
  const Device ParticleStruct*        particlesOld,
  Device uint*                        gridParticleCellIndexNew,
  const Device uint*                  gridParticleCellIndexOld,
  const Device uint*                  gridCellParticleIndices,
  constantKernelInput(uint,           nodeCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  uint particleIndex = threadIndex();

  if (particleIndex >= nodeCount)
  {
    return;
  }

  particleIndex = gridCellParticleIndices[threadIndex()];

  gridParticleCellIndexNew[threadIndex()] = gridParticleCellIndexOld[particleIndex];
  particlesNew[threadIndex()] = particlesOld[particleIndex];
}

#define GRID_SOLVER_NEIGHBOUR_PARTICLE_LOOP_BEGIN \
  GRID_SOLVER_NEIGHBOUR_LOOP_BEGIN \
    const uint2 indexRange = getRangeFromOffset(gridCellParticleOffsets, gridCellIndex); \
    /* iterate over particles in neighboring cells*/ \
    for (int otherNodeIndex = indexRange.x; otherNodeIndex < indexRange.y; otherNodeIndex++) \
    {

#define GRID_SOLVER_PACKED_NEIGHBOUR_PARTICLE_LOOP_BEGIN \
  GRID_SOLVER_PACKED_NEIGHBOUR_LOOP_BEGIN \
    /* iterate over particles in neighboring cells*/ \
    for (int otherNodeIndex = indexRange.x; otherNodeIndex < indexRange.y; otherNodeIndex++) \
    {

#define GRID_SOLVER_BOUNDARY_NEIGHBOUR_PARTICLE_LOOP_BEGIN \
  GRID_SOLVER_BOUNDARY_NEIGHBOUR_LOOP_BEGIN \
    const uint2 indexRange = getRangeFromOffset(boundaryGridCellParticleOffsets, gridCellIndex); \
    /* iterate over particles in neighboring cells*/ \
    for (int otherNodeIndex = indexRange.x; otherNodeIndex < indexRange.y; otherNodeIndex++) \
    {

#define GRID_SOLVER_NEIGHBOUR_PARTICLE_LOOP_END \
    } \
  GRID_SOLVER_NEIGHBOUR_LOOP_END

inline float calculateParticleDensity(
  const ParticleStruct          selfParticle,
#ifdef GRID_SOLVER_HASH_FUNCTION
  const float3                  particleCellPosition,
#endif
  const float                   selfMass,
  const FluidSolverData         fluidSolverData,
  const Device uint*            gridCellParticleOffsets,
  const Device ParticleStruct*  particles,
  const uint                    gridCellIndex,
  const ushort                  gridSize,
  const ushort                  gridSizeExp)
{
  float density = 0.f;
  const float fluidKernelRadiusSq = sqr(fluidSolverData.fluidKernelRadius);

  GRID_SOLVER_NEIGHBOUR_PARTICLE_LOOP_BEGIN
    // iterate over each particle in the cell
    const float3 collisionVector = selfParticle.position - particles[otherNodeIndex].position;
    const float actualDistanceSq = lengthSq(collisionVector);

    density += select(0.f, poly6FunctionVariableSquares(actualDistanceSq, fluidKernelRadiusSq), actualDistanceSq < fluidKernelRadiusSq);
  GRID_SOLVER_NEIGHBOUR_PARTICLE_LOOP_END

  return density * fluidSolverData.fluidKernelFunctionConstant[0] * selfMass;
}

inline float calculateBoundaryParticleDensity(
  const ParticleStruct                selfParticle,
#ifdef GRID_SOLVER_HASH_FUNCTION
  const float3                        particleCellPosition,
#endif
  const FluidSolverData               fluidSolverData,
  const Device uint*                  boundaryGridCellParticleOffsets,
  const Device ParticleStruct*        boundaryParticles,
  const Device ParticleCouplingData*  boundaryParticleCouplingData,
  const uint                          gridCellIndex,
  const ushort                        gridSize,
  const ushort                        gridSizeExp)
{
  float density = 0.f;
  const float fluidKernelRadiusSq = sqr(fluidSolverData.fluidKernelRadius);

  GRID_SOLVER_BOUNDARY_NEIGHBOUR_PARTICLE_LOOP_BEGIN
    // iterate over each particle in the cell
    const float3 collisionVector = selfParticle.position - boundaryParticles[otherNodeIndex].position;
    const float actualDistanceSq = lengthSq(collisionVector);

    density += select(0.f, boundaryParticleCouplingData[otherNodeIndex].volume * poly6FunctionVariableSquares(actualDistanceSq, fluidKernelRadiusSq), actualDistanceSq < fluidKernelRadiusSq);
  GRID_SOLVER_NEIGHBOUR_PARTICLE_LOOP_END

  return density * fluidSolverData.fluidKernelFunctionConstant[0];
}

/*
@kernel Calculate fluid particle density.
@param particlesDensity Calculated particle density.
@param gridCellParticleOffsets Starting offset for each grid cell.
@param gridParticleCellIndex Computed cell index for each particle.
@param particlesPredicted Integrated particle position.
@param boundaryGridCellParticleOffsets Starting offset for each grid cell for boundary particles.
@param particleCouplingData Particle coupling data.
@param boundaryParticles Boundary particles.
@param particleSharedData Particle entity shared data.
@param systemBoundingBox Physics system's bounding box.
@param invRadius Inverse of max particle radius in the system.
@param gridSize Size of grid in one dimension.
@param gridSizeExp Grid size in power of 2.
@param nodeCount Total nodes in the solver.
*/
//#autoArgumentBuffer
Kernel void calculateDensity(
  Device float*                       particlesDensity,
  const Device uint*                  gridCellParticleOffsets,
  const Device uint*                  gridParticleCellIndex,
  const Device ParticleStruct*        particlesPredicted,
  const Device uint*                  boundaryGridCellParticleOffsets,
  const Device ParticleCouplingData*  boundaryParticleCouplingData,
  const Device ParticleStruct*        boundaryParticles,
  const Device ParticleSharedData*    particleSharedData,
  Const XAB*                          systemBoundingBox,
  Const float*                        invRadius,
  constantKernelInput(ushort,         gridSize),
  constantKernelInput(ushort,         gridSizeExp),
  constantKernelInput(uint,           nodeCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint particleIndex = threadIndex();

  if (particleIndex >= nodeCount)
  {
    return;
  }

  const uint gridCellIndex = gridParticleCellIndex[particleIndex];
  DECLARE_SELF_PARTICLE(particlesPredicted, identity, nodeIdentity)
  const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];

#ifdef GRID_SOLVER_HASH_FUNCTION
  const float3 particleCellPosition = (selfParticle.position - systemBoundingBox->min) * invRadius[0];
  float density = calculateParticleDensity(selfParticle, particleCellPosition, 1.f/sharedData.sharedInvMass, sharedData.fluidSolverData, gridCellParticleOffsets, particlesPredicted, gridCellIndex, gridSize, gridSizeExp);
  density += calculateBoundaryParticleDensity(selfParticle, particleCellPosition, sharedData.fluidSolverData, boundaryGridCellParticleOffsets, boundaryParticles, boundaryParticleCouplingData, gridCellIndex, gridSize, gridSizeExp) * 1.f/sharedData.invRestDensity;
#else
  float density = calculateParticleDensity(selfParticle, 1.f/sharedData.sharedInvMass, sharedData.fluidSolverData, gridCellParticleOffsets, particlesPredicted, gridCellIndex, gridSize, gridSizeExp);
  density += calculateBoundaryParticleDensity(selfParticle, sharedData.fluidSolverData, boundaryGridCellParticleOffsets, boundaryParticles, boundaryParticleCouplingData, gridCellIndex, gridSize, gridSizeExp) * 1.f/sharedData.invRestDensity;
#endif

  particlesDensity[particleIndex] = density;
}

/*
@kernel Calculate non fluid particle volume as specified in Versatile Rigid-Fluid Coupling for Incompressible SPH.
@param particleCouplingData Calculated particle coupling data.
@param gridCellParticleOffsets Starting offset for each grid cell.
@param gridParticleCellIndex Computed cell index for each particle.
@param particles Particle positions.
@param systemBoundingBox Physics system's bounding box.
@param invRadius Inverse of max particle radius in the system.
@param gridSize Size of grid in one dimension.
@param gridSizeExp Grid size in power of 2.
@param nodeCount Total nodes in the solver.
*/
Kernel void calculateCouplingData(
  Device ParticleCouplingData*        boundaryParticleCouplingData,
  const Device uint*                  gridCellParticleOffsets,
  const Device uint*                  gridParticleCellIndex,
  const Device uint*                  gridCellParticleIndices,
  const Device ParticleStruct*        particles,
  Const XAB*                          systemBoundingBox,
  Const float*                        invRadius,
  constantKernelInput(FluidSolverData,fluidSolverData),
  constantKernelInput(ushort,         gridSize),
  constantKernelInput(ushort,         gridSizeExp),
  constantKernelInput(uint,           nodeCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  uint particleIndex = threadIndex();

  if (particleIndex >= nodeCount)
  {
    return;
  }

  const uint gridCellIndex = gridParticleCellIndex[particleIndex];
  DECLARE_SELF_PARTICLE(particles, identity, nodeIdentity)

  particleIndex = gridCellParticleIndices[particleIndex];

#ifdef GRID_SOLVER_HASH_FUNCTION
  const float3 particleCellPosition = (selfParticle.position - systemBoundingBox->min) * invRadius[0];
  boundaryParticleCouplingData[particleIndex].volume = 1.f/calculateParticleDensity(selfParticle, particleCellPosition, 1.f, fluidSolverData, gridCellParticleOffsets, particles, gridCellIndex, gridSize, gridSizeExp);
#else
  boundaryParticleCouplingData[particleIndex].volume = 1.f/calculateParticleDensity(selfParticle, 1.f, fluidSolverData, gridCellParticleOffsets, particles, gridCellIndex, gridSize, gridSizeExp);
#endif
}

//#autoArgumentBuffer
Kernel void createBoundaryGridCellHistogram(
  atomicKernelInput(uint,       boundaryGridCellIndexCount),
  Device uint*                  boundaryGridParticleCellIndex,
  const Device ParticleStruct*  particles,
  Const PhySystemSettings*      systemSettings,
  Const XAB*                    systemBoundingBox,
  Const float*                  invRadius,
  constantKernelInput(uint,     nodeCount),
  constantKernelInput(ushort,   gridSize),
  constantKernelInput(ushort,   gridSizeExp)
  KERNEL_GLOBAL_ARGUMENTS)
{
  uint index = threadIndex();

  if (index < nodeCount)
  {
    // if index is in range of fluid nodes
    const PhySystemOffsets phySystemOffsets = systemSettings->globalOffsets[SOLVER_FLUID];
    // offset it to go to next solver node indices
    if (index >= phySystemOffsets.globalNodeOffset)
    {
      index += systemSettings->globalOffsets[SOLVER_FLUID+1].globalNodeOffset - phySystemOffsets.globalNodeOffset;
    }
    const ParticleStruct selfParticle = particles[index];

#ifndef GRID_SOLVER_HASH_FUNCTION
    const float3 inverseMergedBoxSize = ((float)gridSize) / max(gridSize / invRadius[0], systemBoundingBox->max - systemBoundingBox->min);
    int3 quantizedPosition = convertInt3((selfParticle.position - systemBoundingBox->min) * inverseMergedBoxSize);
#else
    int3 quantizedPosition = positionHashFunction((selfParticle.position - systemBoundingBox->min) * invRadius[0], gridSize, gridSizeExp);
#endif

    uint gridCountOffset = -1;
    const ushort solverType = getSolverType(selfParticle.identity);

    if (quantizedPosition.x >= -1 && quantizedPosition.x <= gridSize &&
        quantizedPosition.y >= -1 && quantizedPosition.y <= gridSize &&
        quantizedPosition.z >= -1 && quantizedPosition.z <= gridSize)
    {
      quantizedPosition = quantizedPosition & constructInt3(gridSize-1);
      gridCountOffset = encodeGridIndexInt3(quantizedPosition, gridSizeExp);

      atomicAdd(&boundaryGridCellIndexCount[gridCountOffset], 1);
    }

    boundaryGridParticleCellIndex[threadIndex()] = gridCountOffset;
  }
}

//#autoArgumentBuffer
Kernel void createBoundaryGridCellArrays(
  Device uint*                  boundaryGridCellParticleIndices,
  atomicKernelInput(uint,       systemGridCellParticleOffsets),
  const Device uint*            boundaryGridParticleCellIndex,
  constantKernelInput(uint,     nodeCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index < nodeCount)
  {
    const uint gridCountOffset = boundaryGridParticleCellIndex[index];
    if (gridCountOffset != -1)
    {
      const uint offset = atomicAdd(&systemGridCellParticleOffsets[gridCountOffset], 1);
      boundaryGridCellParticleIndices[offset] = index;
    }
  }
}

//#autoArgumentBuffer
Kernel void reorderBoundaryParticles(
  Device ParticleStruct*              particlesNew,
  const Device ParticleStruct*        particlesOld,
  Device ParticleDifferential*        particleDiffNew,
  const Device ParticleDifferential*  particleDiffOld,
  Device ParticleCouplingData*        boundaryParticleCouplingDataNew,
  const Device ParticleCouplingData*  boundaryParticleCouplingDataOld,
  Device uint*                        boundaryGridParticleCellIndexNew,
  const Device uint*                  boundaryGridParticleCellIndexOld,
  const Device uint*                  boundaryGridCellParticleIndices,
  Device uint*                        boundaryGridParticleSystemIndex,
  Const PhySystemSettings*            systemSettings,
  constantKernelInput(uint,           nodeCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  if (threadIndex() >= nodeCount)
  {
    return;
  }

  uint particleIndex = boundaryGridCellParticleIndices[threadIndex()];

  boundaryGridParticleCellIndexNew[threadIndex()] = boundaryGridParticleCellIndexOld[particleIndex];
  particlesNew[threadIndex()] = particlesOld[particleIndex];
  particleDiffNew[threadIndex()] = particleDiffOld[particleIndex];
  boundaryParticleCouplingDataNew[threadIndex()] = boundaryParticleCouplingDataOld[particleIndex];

  // if index is in range of fluid nodes
  const PhySystemOffsets phySystemOffsets = systemSettings->globalOffsets[SOLVER_FLUID];

  // offset it to go to next solver node indices
  if (particleIndex >= phySystemOffsets.globalNodeOffset)
  {
    particleIndex += systemSettings->globalOffsets[SOLVER_FLUID+1].globalNodeOffset - phySystemOffsets.globalNodeOffset;
  }
  boundaryGridParticleSystemIndex[threadIndex()] = particleIndex;
}

inline float3 surfaceTensionAkinci(const float3 collisionVector, const float actualDistance, const float selfParticleMass, const float otherParticleMass, const ParticleSharedData sharedData)
{
  float h = sharedData.fluidSolverData.fluidKernelRadius;
  float h_3 = h * h * h;
  float h_6 = h_3 * h_3;
  float h_9 = h_6 * h_3;
  float x = (h - actualDistance);
  x = x * x * x;
  float y = x * actualDistance * actualDistance * actualDistance;
  return collisionVector * (selfParticleMass * otherParticleMass * sharedData.surfaceTensionCoeff * (32.f / (3.1415926535897932384626433832795f * h_9 * actualDistance)) * select(2.f * y - h_6 / 64.f, y, (2.f * actualDistance) > h));
}

/*inline float poly6Function(const float r, const float h)
{
  const float x = (h * h - r * r) / (h * h * h);
  return (315.f / (64.f * M_PI_F)) * x * x * x;
}

inline float poly6FunctionGradient(const float r, const float h)
{
  const float invH = 1.f/h;
  const float x = (h * h - r * r) * sqr(sqr(invH * invH));
  return (-945.f / (32.f * M_PI_F)) * x * x * invH;
}

inline float poly6FunctionLaplacian(const float r, const float h)
{
  const float x = (h * h - r * r) * (-7 * r * r + 3 * h * h) / (h * sqr(sqr(h * h)));
  return (945.f / (32.f * M_PI_F)) * x;
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

inline float viscosityFunctionLaplacian(const float r, const float h)
{
  const float x = (h - r) / sqr(h * h * h);
  return (45.f / M_PI_F) * x;
}*/

#endif

#endif
