#ifndef COLLISION_SOLVER_SHADER
#define COLLISION_SOLVER_SHADER

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
#ifdef SET_PARTICLE_BOUNDING_BOXES
  Device XAB*                       particleBoundingBoxes,
#endif
  Device XAB*                       particleGroupBoundingBoxes,
  const Device ParticleStruct*      particlesPredicted,
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
  // bounding box for the batch
  XAB accumulatedBoundingBox;

  // reset to INF, -INF
  clearXAB(&accumulatedBoundingBox, INFINITY);

  for (uint index = threadIndex(); index < nodeCount; index += threadGroupCount() * threadGroupSize())
  {
    const ParticleStruct particle = particlesPredicted[index];
    ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(particle.identity);
    const PhySystemOffsets phySystemOffsets = globalOffsets[nodeIdentity.solverType];

    nodeIdentity.entityId += phySystemOffsets.globalSolverOffset;
    nodeIdentity.instanceId += phySystemOffsets.globalInstanceOffset;

    const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];
    const ParticleNodeLocator nodeLocator = getNodeLocator(index, phySystemOffsets.globalNodeOffset + partitions[nodeIdentity.instanceId].offset, entityLocation[nodeIdentity.entityId].node);

    const float radius = getRadiusUsingDeviceAux(&sharedData, particleAuxData, nodeLocator.commonNodeIndex);

    XAB particleBoundingBox;
    particleBoundingBox.min = particle.position - constructFloat3(radius);
    particleBoundingBox.max = particle.position + constructFloat3(radius);

#ifdef SET_PARTICLE_BOUNDING_BOXES
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
