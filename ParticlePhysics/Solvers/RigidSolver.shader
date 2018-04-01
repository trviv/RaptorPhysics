#ifndef RIGID_SOLVER_SHADER
#define RIGID_SOLVER_SHADER

/*
@kernel Compute covariance matrix.
@param matrixData Matrix data output.
@param particleDeltas Change in particle position.
@param particlesPredicted Current particle position.
@param particleIdentities Particle identifiers.
@param particlesTemp Current particle position.
@param rigidBodyData Rigid body data.
@param partitions Instance partition data.
@param sectionData Entity section data.
@param length Rigid body count.
*/
Kernel void covarianceMatrix(
  Device float*                   matrixData,
  Device ParticleStruct*          particleDeltas,
  const Device ParticleStruct*    particlesPredicted,
  const Device IdentityInfo*      particleIdentities,
  const Device ParticleStruct*    particlesTemp,
  const Device ParticleRigidData* rigidBodyData,
  const Device PartitionInfo*     partitions,
  const Device SectionData*       sectionData,
  const uint length)
{
  const uint index = threadIndex();
  const uint localIndex = threadLocalIndex();

  Shared float3 currentComOffset[COMPUTE_MAX_THREADS];
  Shared float3 initialComOffset[COMPUTE_MAX_THREADS];

  if (index < length)
  {
    const IdentityInfo identity = particleIdentities[index];
    const uint solverId = getSolverId(identity);
    const uint entityId = getEntityId(identity);

    const SectionData localSectionData = sectionData[solverId];

    const uint absoluteNodeOffset = partitions[entityId].offset;
    const uint relativeNodeIndex = index % localSectionData.node.count;
    const uint absoluteNodeIndex = absoluteNodeOffset + relativeNodeIndex;
    const uint commonNodeIndex = localSectionData.node.offset + relativeNodeIndex;

    currentComOffset[localIndex] = particlesPredicted[absoluteNodeIndex].position - particlesTemp[entityId].position;
    initialComOffset[localIndex] = rigidBodyData[commonNodeIndex].initialComOffset;

    // set delta now because com is available, and will be overwritten later
    // refer unified particle physics
    particleDeltas[absoluteNodeIndex].position = -currentComOffset[localIndex];

    const Shared float* currentComOffsetPtr = (currentComOffset + localIndex);
    const Shared float* initialComOffsetPtr = (initialComOffset + localIndex);
    Device float* matrixRow = (matrixData + 9 * absoluteNodeIndex);

    uint offset = 0;
    for (uint i = 0; i < 3; i++)
    {
      for (uint j = 0; j < 3; j++)
      {
        matrixRow[offset] = currentComOffsetPtr[j] * initialComOffsetPtr[i];
        offset++;
      }
    }
  }
}

void setAdjugateMatrix(
  Device float* matrixAdjugateCofactor,
  Device float* matrix2,
  const Device float* matrixData)
{
  const char mod3[5] = { 0, 1, 2, 0, 1 };
  uint offset = 0;

  for (uint i = 0; i < 3; i++)
  {
    for (uint j = 0; j < 3; j++)
    {
      // adjugate is the transpose of cofactor
      // https://en.wikipedia.org/wiki/Adjugate_matrix

      float cellValue =
        matrixData[mod3[i + 1] * 3 + mod3[j + 1]] *
        matrixData[mod3[i + 2] * 3 + mod3[j + 2]] -
        matrixData[mod3[i + 1] * 3 + mod3[j + 2]] *
        matrixData[mod3[i + 2] * 3 + mod3[j + 1]];

      matrix2[offset] = cellValue;
      matrixAdjugateCofactor[offset] = cellValue * matrixData[offset];
      offset++;
    }
  }
}

float getGamma(const Device float* matrix2, const Device float* matrixPtr, const float determinant)
{
  float mat_inf = 0,
    mat_one = 0,
    adj_inf = 0,
    adj_one = 0;

  for (uint x = 0; x < 3; x++)
  {
    float sum_mat_0 = 0,
      sum_mat_1 = 0,
      sum_adj_0 = 0,
      sum_adj_1 = 0;

    for (uint y = 0; y < 3; y++)
    {
      const uint offset0 = (x * 3) + y;
      const uint offset1 = (y * 3) + x;
      sum_mat_0 += fabs(matrixPtr[offset0]);
      sum_mat_1 += fabs(matrixPtr[offset1]);
      sum_adj_0 += fabs(matrix2[offset0]);
      sum_adj_1 += fabs(matrix2[offset1]);
    }

    if (sum_mat_0 > mat_inf)  mat_inf = sum_mat_0;
    if (sum_mat_1 > mat_one)  mat_one = sum_mat_1;
    if (sum_adj_0 > adj_inf)  adj_inf = sum_adj_0;
    if (sum_adj_1 > adj_one)  adj_one = sum_adj_1;
  }

  return sqrt(sqr((adj_one * adj_inf) / (mat_one * mat_inf)) / fabs(determinant));
}

/*
@kernel Matrix SVD decomposition kernel.
@param matrixData Matrix data output.
@param matrixTempInput1 Temporary array 1.
@param matrixTempInput2 Temporary array 2.
@param iterations Iterations for the solver.
@param length Rigid body count.
*/
Kernel void rigidSolver(
  Device float* matrixData,
  Device float* matrixTempInput1,
  Device float* matrixTempInput2,
  const uint iterations,
  const uint length)
{
  uint index = threadIndex();
  const uint grid = groupSize();
  const uint originalIndex = index;

  for (uint it = 0; it < iterations; it++)
  {
    const uint m = 0;
    index = originalIndex + grid*m;

    if (index < length)
    {
      Device float* matrix1;
      Device float* matrix2;
      Device float* matrixPtr = (matrixData + index * 9);

      matrix1 = (it & 1) ? matrixTempInput2 : matrixTempInput1;
      matrix2 = (it & 1) ? matrixTempInput1 : matrixTempInput2;

      matrix1 += index * 9;
      matrix2 += index * 9;

      setAdjugateMatrix(matrix1, matrix2, matrixPtr);

      const float determinant = matrix1[0] + matrix1[1] + matrix1[2];
      const float gamma = getGamma(matrix2, matrixPtr, determinant);
      const float g1 = gamma * .5f;
      const float g2 = .5f / (gamma * determinant);
      uint matIndex = 0;

      for (uint i = 0; i < 3; i++)
      {
        for (uint j = 0; j < 3; j++)
        {
          matrixPtr[matIndex] = g1 * matrixPtr[matIndex] + g2 * matrix2[matIndex];
          matIndex++;
        }
      }
    }
  }
}

/*
@kernel Set delta position for rigid body.
@param particleDeltas Change in particle position.
@param particleIdentities Particle identifiers.
@param matrixData Instance transformation matrix data.
@param rigidBodyData Rigid body data.
@param partitions Instance partition data.
@param sectionData Entity section data.
@param length Rigid body count.
*/
Kernel void setDeltaPosition(
  Device ParticleStruct*          particleDeltas,
  const Device IdentityInfo*      particleIdentities,
  const Device float*             matrixData,
  const Device ParticleRigidData* rigidBodyData,
  const Device PartitionInfo*     partitions,
  const Device SectionData*       sectionData,
  const uint length)
{
  const uint index = threadIndex();
  const uint localIndex = threadLocalIndex();
  Shared float3 initialComOffset[COMPUTE_MAX_THREADS];

  if (index < length)
  {
    const IdentityInfo identity = particleIdentities[index];
    const uint solverId = getSolverId(identity);
    const uint entityId = getEntityId(identity);

    const SectionData localSectionData = sectionData[solverId];

    const uint absoluteNodeOffset = partitions[entityId].offset;
    const uint relativeNodeIndex = index % localSectionData.node.count;
    const uint absoluteNodeIndex = absoluteNodeOffset + relativeNodeIndex;
    const uint commonNodeIndex = localSectionData.node.offset + relativeNodeIndex;

    matrixData += 9 * entityId;

    initialComOffset[localIndex] = rigidBodyData[commonNodeIndex].initialComOffset;

    float3 comOffsetCrossQ;
    Thread float* comOffsetCrossQPtr = (Thread float*)&comOffsetCrossQ;

    for (uint i = 0; i < 3; i++)
    {
      const Device float* particleMatrix = matrixData + i;
      const float3 product = initialComOffset[localIndex] * constructFloat3(particleMatrix[0], particleMatrix[3], particleMatrix[6]);
      comOffsetCrossQPtr[i] = product.x + product.y + product.z;
    }

    particleDeltas[absoluteNodeIndex].position += comOffsetCrossQ;
  }
}

#endif