#include "RigidConstrain.h"

CU_DEV void RigidConstrain::getDelta(ValueType& del, const IndexType index,
  const IndexType connection_index, const IndexType offset,
  const ConstrainBuffer buffer_index)
{
  del = 0;
  real w1 = getMass()[index];
  real w2 = getMass()[connection_index];
  if (w1 + w2 < real(0.0000001)) return;

  del = getValue(index, buffer_index) -
    getValue(connection_index, buffer_index);
  del *= -w1*(real(1) - getDistance()[offset] / del.length()) / (w1 + w2);
}

void RigidConstrain::add(const IndexType index, const IndexType connection,
  const CoefType coef, const real distance,
  const real inv_mass)
{
  Constrain::add(index, connection, coef);
  if (index == connection)
  {
    expand<real>(index + offset, point_mass);
    point_mass[index + offset] = inv_mass;
  }
  //else
  //{
  expand<std::vector<real>>(index + offset, point_distance);
  point_distance[index + offset].push_back(distance);
  //}
}

void RigidConstrain::exportToDevice(__int8** device_additional_memory,
  Counter additional_size, Counter baseSize)
{
  __int8* device_memory = NULL;

  Real3 com(0);
  std::vector<Real3> com_offset;
  for (Counter i = 0; i < constrain_values.size(); i++)
    com += constrain_values[i];
  com /= real(constrain_values.size());
  for (Counter i = 0; i < constrain_values.size(); i++)
    com_offset.push_back(constrain_values[i] - com);

  IndexType total_size = sizeof(Real3)*com_offset.size() +
    9 * sizeof(real)*constrain_values.size() +
    sizeof(Real3)*constrain_values.size();

  constrain_values[0] -= real(.5);
  DistanceConstrain::exportToDevice(&device_memory, total_size,
    sizeof(RigidConstrain));

  device_com_offset = (Real3*)device_memory;
  device_matrix = (real*)(device_com_offset + com_offset.size());
  device_del_pos = (Real3*)(device_matrix + 9 * constrain_values.size());

  DeviceEntity<Real3>::exportToDevice(&com_offset[0], device_com_offset,
    com_offset.size());
  DeviceEntity<real>::set(device_matrix, 0, 9 * getNodeCount());
  DeviceEntity<Real3>::set(device_del_pos, 0, getNodeCount());
  DeviceEntity<RigidConstrain>::exportToDevice(this,
    (RigidConstrain*)constrain_alloc);

  std::cout << "Rigid Alloc:\n";
  std::cout << "Base: " << baseSize <<
    "\tAddr: " << (__int64)constrain_alloc << "\n";
  std::cout << "Com offset: " <<
    sizeof(Real3)*com_offset.size() <<
    "\tAddr: " << (__int64)device_com_offset << "\n";
  std::cout << "Matrix: " <<
    9 * sizeof(real)*constrain_values.size() <<
    "\tAddr: " << (__int64)device_matrix <<
    "\tDiff: " << (__int64)device_matrix - (__int64)device_com_offset << "\n";
  std::cout << "Del pos: " <<
    sizeof(Real3)*constrain_values.size() <<
    "\tAddr: " << (__int64)device_del_pos <<
    "\tDiff: " << (__int64)device_del_pos - (__int64)device_matrix << "\n";
}

CU_KER void covarianceMatrix(
  RigidConstrain* constrain)
{
  Counter index = threadIndex;
  const Counter i = ((index % 9) / 3), j = ((index % 9) % 3);
  index = index / 9;
  if (index < constrain->getNodeCount())
  {
    const real new_com_offset = constrain->getPosition()[index][j] -
      constrain->getValue(0, VAR1)[j];
    constrain->getMatrix()[index + ((3 * i) + j)*constrain->getNodeCount()]
      = new_com_offset * constrain->getComOffset()[index][i];// / constrain->getNodeCount();
    /*
    int ind = index + ((3 * i) + j)*constrain->getNodeCount();
    constrain->getMatrix()[ind] = 0;
    for (int k = 0; k < constrain->getNodeCount(); k++)
    {
    constrain->getMatrix()[ind] = new_com_offset[i] - constrain->getValue(;
    }*/

    //= diff[i] * diff[j]
    //constrain->getMatrix()[index + 3 * i] = 0;
    if (j == 0)
    {
      constrain->getDelPosition()[index][i] = constrain->getValue(0, VAR1)[i] -
        constrain->getPosition()[index][i];
    }
    /*
    debugLog("Main Index: %d\n\t%f\n",
    index + ((3 * i) + j)*constrain->getNodeCount(),
    constrain->getMatrix()[index + ((3 * i) + j)*constrain->getNodeCount()]);
    */
    if (i == 0 && j == 0)
    {
      /*
      debugLog("Index: %d\n\tCom offset: %f, %f, %f\n\tDel pos: %f, %f, %f\n", index,
      constrain->getMatrix()[index + ((3 * i) + j)*constrain->getNodeCount()],
      constrain->getComOffset()[index][0],
      constrain->getComOffset()[index][1],
      constrain->getComOffset()[index][2],
      constrain->getDelPosition()[index][0],
      constrain->getDelPosition()[index][1],
      constrain->getDelPosition()[index][2]);
      */
    }
    //constrain->getDelPosition()[index] = 0;
  }
}

CU_KER void svdKernel(
  RigidConstrain* constrain,
  const Counter iterations)
{
  Counter index = threadIndex;
  //const Counter grid = blockDim.x*blockDim.y;
  //const Counter multiplier = 1;// Counter(mCeil(length / real(grid)));
  const Counter backIndex = index;
  const Counter length = 9;
  for (Counter it = 0; it < iterations; it++)
  {
    //for (Counter m = 0; m < multiplier; m++)
    {
      //index = backIndex + grid*m;
      index = backIndex;
      const Counter i = ((index % 9) / 3), j = ((index % 9) % 3);
      index = index / 9;
      if (index < length)
      {
        real adjTra;
        const real* value = (constrain->getMatrix() + index);
        adjTra = value[((i + 1) % 3) * 3 + ((j + 1) % 3)] *
          value[((i + 2) % 3) * 3 + ((j + 2) % 3)] -
          value[((i + 1) % 3) * 3 + ((j + 2) % 3)] *
          value[((i + 2) % 3) * 3 + ((j + 1) % 3)];

        ((real*)constrain->getValueBuffer(VAR1))[index + i * 3 + j] = adjTra;
        ((real*)constrain->getValueBuffer(VAR0))[index + i * 3 + j] =
          adjTra*value[i * 3 + j];
        //printf("%f ", ((real*)constrain->getValueBuffer(VAR1))[index + i * 3 + j]);
        //if (i == 0 && j == 0) printf("\n");
      }
    }
    __syncthreads();

    //for (Counter m = 0; m < multiplier; m++)
    {
      //index = backIndex + grid*m;
      index = backIndex;
      const Counter i = ((index % 9) / 3), j = ((index % 9) % 3);
      index = index / 9;
      if (index < length)
      {
        real det = ((real*)constrain->getValueBuffer(VAR0))[index] +
          ((real*)constrain->getValueBuffer(VAR0))[index + 1] +
          ((real*)constrain->getValueBuffer(VAR0))[index + 2];
        real gamma;
        {
          real mat_inf = 0, mat_one = 0, adj_inf = 0, adj_one = 0;
          for (Counter x = 0; x < 3; x++)
          {
            real sum_mat_1 = 0, sum_adj_1 = 0,
              sum_mat_0 = 0, sum_adj_0 = 0;
            for (Counter y = 0; y < 3; y++)
            {
              sum_mat_0 += fabs(constrain->getMatrix()[index + (x * 3) + y]);
              sum_mat_1 += fabs(constrain->getMatrix()[index + (y * 3) + x]);
              sum_adj_0 += fabs(((real*)constrain->getValueBuffer(VAR1))
                [index + (x * 3) + y]);
              sum_adj_1 += fabs(((real*)constrain->getValueBuffer(VAR1))
                [index + (y * 3) + x]);
            }
            if (sum_mat_0 > mat_inf)  mat_inf = sum_mat_0;
            if (sum_mat_1 > mat_one)  mat_one = sum_mat_1;
            if (sum_adj_0 > adj_inf)  adj_inf = sum_adj_0;
            if (sum_adj_1 > adj_one)  adj_one = sum_adj_1;
          }
          gamma = mSqrt(mSqr((adj_one*adj_inf) / (mat_one*mat_inf))
            / fabs(det));
        }
        const real g1 = gamma*real(.5);
        const real g2 = real(.5) / (gamma*det);
        const Counter mat_index = index + (i * 3) + j;
        constrain->getMatrix()[mat_index] =
          g1*constrain->getMatrix()[mat_index] +
          g2*((real*)constrain->getValueBuffer(VAR1))[mat_index];
      }
    }
    __syncthreads();
  }
}

CU_KER void rigidSolver(
  RigidConstrain* constrain,
  const Counter step,
  const Counter length)
{
  Counter index = threadIndex;
  const Counter grid = blockDim.x*blockDim.y;
  const Counter multiplier = 1;// Counter(mCeil(length / real(grid)));
  const Counter backIndex = index;
  for (Counter it = 0; it < step; it++)
  {
    for (Counter m = 0; m < multiplier; m++)
    {
      index = backIndex + grid*m;
      const Counter i = ((index % 9) / 3), j = ((index % 9) % 3);
      index = index / 9;
      if (index < length)
      {
        real adjTra;
        const real* value = (constrain->getMatrix() + index);
        adjTra = value[((i + 1) % 3) * 3 + ((j + 1) % 3)] *
          value[((i + 2) % 3) * 3 + ((j + 2) % 3)] -
          value[((i + 1) % 3) * 3 + ((j + 2) % 3)] *
          value[((i + 2) % 3) * 3 + ((j + 1) % 3)];

        ((real*)constrain->getValueBuffer(VAR1))[index + i * 3 + j] = adjTra;
        ((real*)constrain->getValueBuffer(VAR0))[index + i * 3 + j] =
          adjTra*value[i * 3 + j];
        //printf("%f ", ((real*)constrain->getValueBuffer(VAR1))[index + i * 3 + j]);
        //if (i == 0 && j == 0) printf("\n");
      }
    }
    __syncthreads();

    for (Counter m = 0; m < multiplier; m++)
    {
      index = backIndex + grid*m;
      const Counter i = ((index % 9) / 3), j = ((index % 9) % 3);
      index = index / 9;
      if (index < length)
      {
        real det = ((real*)constrain->getValueBuffer(VAR0))[index] +
          ((real*)constrain->getValueBuffer(VAR0))[index + 1] +
          ((real*)constrain->getValueBuffer(VAR0))[index + 2];
        real gamma;
        {
          real mat_inf = 0, mat_one = 0, adj_inf = 0, adj_one = 0;
          for (Counter x = 0; x < 3; x++)
          {
            real sum_mat_1 = 0, sum_adj_1 = 0,
              sum_mat_0 = 0, sum_adj_0 = 0;
            for (Counter y = 0; y < 3; y++)
            {
              sum_mat_0 += fabs(constrain->getMatrix()[index + (x * 3) + y]);
              sum_mat_1 += fabs(constrain->getMatrix()[index + (y * 3) + x]);
              sum_adj_0 += fabs(((real*)constrain->getValueBuffer(VAR1))
                [index + (x * 3) + y]);
              sum_adj_1 += fabs(((real*)constrain->getValueBuffer(VAR1))
                [index + (y * 3) + x]);
            }
            if (sum_mat_0 > mat_inf)  mat_inf = sum_mat_0;
            if (sum_mat_1 > mat_one)  mat_one = sum_mat_1;
            if (sum_adj_0 > adj_inf)  adj_inf = sum_adj_0;
            if (sum_adj_1 > adj_one)  adj_one = sum_adj_1;
          }
          gamma = mSqrt(mSqr((adj_one*adj_inf) / (mat_one*mat_inf))
            / fabs(det));
        }
        const real g1 = gamma*real(.5);
        const real g2 = real(.5) / (gamma*det);
        const Counter mat_index = index + (i * 3) + j;
        constrain->getMatrix()[mat_index] =
          g1*constrain->getMatrix()[mat_index] +
          g2*((real*)constrain->getValueBuffer(VAR1))[mat_index];
      }
    }
    __syncthreads();
  }
}

CU_KER void deltaPos(RigidConstrain* constrain)
{
  Counter index = threadIndex;
  const Counter i = (index % 3);
  index = index / 3;
  if (index < constrain->getNodeCount())
  {
    //real multiplier = -1;
    /*
    const real det =
    ((real*)constrain->getValueBuffer(VAR0))[index / constrain->getNodeCount()] +
    ((real*)constrain->getValueBuffer(VAR0))[index / constrain->getNodeCount() + 1] +
    ((real*)constrain->getValueBuffer(VAR0))[index / constrain->getNodeCount() + 2];
    //if (det > 0)
    */
    //multiplier = 1;
    /*
    real* mat = constrain->getMatrix() + i * 3;
    constrain->getPosition(index)[i] +=
    //constrain->getValue(index, VAR1)[i] +=
    constrain->getDelPosition(index)[i]
    + (constrain->getComOffset()[index][0] * mat[i] +
    constrain->getComOffset()[index][1] * mat[i + 1] +
    constrain->getComOffset()[index][2] * mat[i + 2]);
    */

    real* mat = constrain->getMatrix() + i;
    debugLog("Mat:%f %f %f\n", mat[0], mat[3], mat[6]);
    real com_offset_cross_q =
      constrain->getComOffset()[index][0] * mat[0] +
      constrain->getComOffset()[index][1] * mat[3] +
      constrain->getComOffset()[index][2] * mat[6];

    /*
    real* mat = constrain->getMatrix() + i * 3;
    real com_offset_cross_q =
    constrain->getComOffset()[index][0] * mat[i] +
    constrain->getComOffset()[index][1] * mat[i + 1] +
    constrain->getComOffset()[index][2] * mat[i + 2];
    */
    //constrain->getPosition()[index][i] +=
    //constrain->getDelPosition()[index][i] + com_offset_cross_q;

    constrain->getDelPosition()[index][i] += com_offset_cross_q;

    constrain->getPosition()[index][i] += constrain->getDelPosition()[index][i];
    constrain->getValueBuffer(DEF)[index][i] += constrain->getDelPosition()[index][i];
    //constrain->getDelPosition()[index][i] = constrain->getPosition()[index][i] - constrain->getValueBuffer(DEF)[index][i];

    //constrain->getDelPosition()[index][i] *= -1;
    //constrain->getDelPosition()[index][i] = -constrain->getDelPosition()[index][i];

    /*
    printf("Del: %f %f\n", constrain->getDelPosition(index)[i],
    (constrain->getComOffset()[index][0] * mat[i] +
    constrain->getComOffset()[index][1] * mat[i + 1] +
    constrain->getComOffset()[index][2] * mat[i + 2]));
    printf("%f %f\n", constrain->getPosition(index)[i],
    constrain->getDelPosition(index)[i]
    + (constrain->getComOffset()[index][0] * mat[i] +
    constrain->getComOffset()[index][1] * mat[i + 1] +
    constrain->getComOffset()[index][2] * mat[i + 2]));
    */
    /*
    printf("Del:%f %f\n", constrain->getDelPosition(index)[i],
    (constrain->getComOffset()[index][0] * mat[0] +
    constrain->getComOffset()[index][1] * mat[3] +
    constrain->getComOffset()[index][2] * mat[6]));
    */
    /*
    debugLog("Del:%d %f %f\n", index * 3 + i,
    constrain->getDelPosition()[index][i], com_offset_cross_q);
    debugLog("%d %f %f\n", index * 3 + i, constrain->getPosition()[index][i],
    constrain->getDelPosition()[index][i] + com_offset_cross_q);
    */
  }
}

void RigidConstrain::solve()
{
  DeviceEntity<Real3>::copy(getValueBuffer(DEF), getPosition(), getNodeCount());
  integrate();

  showMatrix(3, this->getNodeCount(), (real*)this->getVelocity());
  showMatrix(3, this->getNodeCount(), (real*)this->getForce());
  showMatrix(3, this->getNodeCount(), (real*)this->getPosition());
  dim3 threads_pre, blocks_pre;
  dim3 threads_post, blocks_post;
  dim3 threads_upd, blocks_upd;

  configureGrid(blocks_pre, threads_pre, getNodeCount() * 9);
  configureGrid(blocks_post, threads_post, 9);
  configureGrid(blocks_upd, threads_upd, getNodeCount() * 3);

  //for (Counter i = 0; i < iterations; i++)
  {
    //copy to aux buffer to find new COM
    DeviceEntity<Real3>::copy(getValueBuffer(VAR1), getPosition(), getNodeCount());
    //DeviceEntity<Real3>::copy(getValueBuffer(DEF), getPosition(), getNodeCount());
    mean<Real3>(getValueBuffer(VAR1), getNodeCount());
    debugLog("Mean:\n");
    showMatrix(1, 3, (real*)getValueBuffer(VAR1));
    cudaDeviceSynchronize();
    CU_PROMPT;

    showMatrix(3, getNodeCount(), (real*)this->device_com_offset);
    covarianceMatrix << <blocks_pre, threads_pre >> >((RigidConstrain*)constrain_alloc);
    cudaDeviceSynchronize();
    CU_PROMPT;

    // add all n * 9 values to form = 3x3 matrix
    for (Counter j = 0; j < 9; j++)
    {
      mean<real>(device_matrix + (j*getNodeCount()), getNodeCount());
      // place the value in forst 9 elements
      DeviceEntity<real>::copy(device_matrix + j,
        device_matrix + (j*getNodeCount()));
      CU_PROMPT;
    }
    // make a copy
    DeviceEntity<real>::copy(device_matrix + 9,
      device_matrix, 9);

    debugLog("M:\n");
    showMatrix(3, 3, device_matrix);
    cudaDeviceSynchronize();
    CU_PROMPT;

    rigidSolver << <1, threads_post >> >
      ((RigidConstrain*)constrain_alloc, 16, 1);
    cudaDeviceSynchronize();
    CU_PROMPT;

    debugLog("Q:\n");
    showMatrix(3, 3, device_matrix);
    cudaDeviceSynchronize();
    CU_PROMPT;

    DeviceEntity<Real3>::copy(getValueBuffer(VAR1),
      getPosition(), getNodeCount());

    //DeviceEntity<Real3>::set(constrain_alloc->getDelPosition(), 0, getNodeCount());
    deltaPos << <blocks_upd, threads_upd >> >((RigidConstrain*)constrain_alloc);
    cudaDeviceSynchronize();
    CU_PROMPT;

    //showMatrix(this->getNodeCount() * 3, 1, (real*)this->getDelPosition());
    //getch();
  }

  //DeviceEntity<Real3>::copy(getValueBuffer(DEF), getPosition(), getNodeCount());
  //DistanceConstrain::solve();

  //sum(getPosition(), getValueBuffer(DEF), getPosition(), getNodeCount());
  //getch();
  /*
  if ((getIterations() & 1) == 0)
  {
  DeviceEntity<Real3>::copy((Real3*)getValueBuffer(DEF),
  (Real3*)getValueBuffer(VAR0), getNodeCount());
  cudaDeviceSynchronize();
  }
  */
  //DeviceEntity<Real3>::copy(getPosition(), getValueBuffer(VAR1),
  //getNodeCount());
  //DeviceEntity<Real3>::copy(getValueBuffer(VAR1), getPosition(),
  //getNodeCount());
  differentiate();

  //DeviceEntity<Real3>::copy(getPosition(), getValueBuffer(VAR1),
  //getNodeCount());
  //DeviceEntity<Real3>::copy(getValueBuffer(VAR1), getPosition(),
  //getNodeCount());
}