#include <Core.h>
#include <vector>

#ifndef __CUDA_UTILS
#define __CUDA_UTILS

#define WARP_SIZE 32
#define MAX_BLOCK_PARALLELISM 1024

#define compare(a, b, less) !((a > b) ^ (less))

#define debugLog(a, ...) printf(a, __VA_ARGS__)

#define threadIndex threadIdx.x + threadIdx.y*blockDim.x + (blockIdx.x + blockIdx.y*gridDim.x)*blockDim.x*blockDim.y

/// function to perform bitonic sort
template<class T>CU_KER void bitonicSort(T* series, Counter length)
{
  extern __shared__ T shared[];

  // the number of threads for the kernel
  const Counter threads = blockDim.x*blockDim.y;

  // the number of times the block should execute
  const Counter multiplier = (Counter)mCeil(length / float(threads << 1));

  // the upper exponent of 2 according to length
  Counter block_size = mExpOf2(length);
  if (((length - 1)&length) > 0) block_size++;

  // the execution slot for this thread
  const Counter thread_index = threadIdx.x + threadIdx.y*blockDim.x;
  const Counter thread_index_2 = (thread_index << 1);

  // maximum times block has to iterate
  // upper limit is 5 which goes well with shared memory
  Counter max_block;
  {
    Counter shared_pow = mExpOf2(threads);
    max_block = 5;
    max_block = max_block < shared_pow ? max_block : shared_pow;
  }

  for (Counter i = 0; i < block_size; i++)  //iterate over exponent of 2's
  {
    Counter k;
    Counter block_mask = (1 << (i + 1));

    for (k = i; k > max_block; k--) //sort more the 64 elements
    {
      const Counter pow_k = (1 << k);
      for (Counter m = 0; m < multiplier; m++)
      {
        Counter index = m*threads + thread_index;
        index = (index&(pow_k - 1)) + (pow_k + pow_k)*(index >> k);
        const Counter index_swap = index + pow_k;

        if (index_swap < length)
        {
          shared[thread_index_2] = series[index];
          shared[thread_index_2 + 1] = series[index_swap];
          if (compare(shared[thread_index_2], shared[thread_index_2 + 1],
            (index&block_mask) == 0))
          {
            series[index] = shared[thread_index_2 + 1];
            series[index_swap] = shared[thread_index_2];
          }
        }
      }
    }

    for (Counter m = 0; m < multiplier; m++)
    {
      const Counter offset = ((m*threads + thread_index) << 1);
      if ((offset + 1) < length)
      {
        shared[thread_index_2] = series[offset];
        shared[thread_index_2 + 1] = series[offset + 1];

        for (Counter k1 = k; k1 >= 0; k1--)
        {
          const Counter pow_k = (1 << k1);
          Counter index = thread_index;
          index = (index&(pow_k - 1)) + (pow_k + pow_k)*(index >> k1);
          Counter absolute_index = (offset >> 1);
          absolute_index = (absolute_index&(pow_k - 1)) + (pow_k + pow_k)*
            (absolute_index >> k1);
          const Counter index_swap = index + pow_k;
          if (compare(shared[index], shared[index_swap],
            (absolute_index&block_mask) == 0))
          {
            T temp = shared[index];
            shared[index] = shared[index_swap];
            shared[index_swap] = temp;
          }
        }

        series[offset] = shared[thread_index_2];
        series[offset + 1] = shared[thread_index_2 + 1];
      }
    }
    __syncthreads();
  }
  for (Counter m = 0; m < multiplier; m++)
  {
    Counter index = m*threads + thread_index;
    if (index > 0 & index < length)
    {
      if (series[index] < series[index - 1])
      {
        printf("Error %d %f %f\n", index, series[index], series[index - 1]);
      }
    }
  }
}

//template CU_KER void bitonicSort<real>(real* series, Counter length);

template<class T> void sort(T* host_array, const Counter count)
{
  float *device_array = NULL;
  cudaMalloc(&device_array, sizeof(T)*count);
  CU_PROMPT;
  cudaMemcpy(device_array, host_array, sizeof(T)*count,
    cudaMemcpyHostToDevice);
  CU_PROMPT;
  std::cout << "start" << "\n";
  Counter y = count / 64 > 32 ? 32 : count / 64;
  bitonicSort << < 1, dim3(32, y), 2 * 32 * y*sizeof(float) >> >
    (device_array, count);
  CU_PROMPT;
  cudaDeviceSynchronize();
  std::cout << "stop" << "\n";
  CU_PROMPT;
  cudaMemcpy(host_array, device_array, sizeof(float)*count,
    cudaMemcpyDeviceToHost);
  CU_PROMPT;
  cudaFree(device_array);
  CU_PROMPT;
}

static CU_HOST void configureGrid(dim3& blocks, dim3& threads,
  const Counter elements)
{
  threads = dim3(WARP_SIZE, (Counter)mCeil(float(elements) / WARP_SIZE), 1);
  if (threads.y > WARP_SIZE)
  {
    blocks.x = (Counter)mCeil(threads.y / float(WARP_SIZE));
    threads.y = WARP_SIZE;
  }
  if (threads.y == 0) threads.y = 1;
}

template<class T>CU_KER void meanKernel(T* device_array, const Counter length,
  const Counter iteration, const Counter max_iterations, const Counter divide)
{
  Counter index = ((threadIdx.x + threadIdx.y*blockDim.x +
    (blockIdx.x + blockIdx.y*gridDim.x)*blockDim.x*blockDim.y) <<
    (1 + iteration));
  for (Counter i = 0; i < max_iterations; i++)
  {
    Counter index2 = index + (1 << (i + iteration));
    if (index2 < length) device_array[index] += device_array[index2];
    index <<= 1;
    __syncthreads();
  }
  if (index == 0 && divide) device_array[index] /= length;
}

template<class T>void mean(T* device_array, const Counter length,
  const bool only_sum = false)
{
  Counter iterations = mExpOf2(length);
  Counter max_block_parallelism = mExpOf2(MAX_BLOCK_PARALLELISM << 1);
  for (Counter i = 0; i < iterations; i++)
  {
    dim3 threads, blocks;
    configureGrid(blocks, threads, Counter(mCeil(float(length) / ((1 << i) * 2))));
    meanKernel << <blocks, threads >> >(device_array, length, i,
      ((iterations - i)>max_block_parallelism) ? 1 : (iterations - i),
      ((iterations - i) <= max_block_parallelism) ? (only_sum ? 0 : 1) : 0);
    cudaDeviceSynchronize();
    CU_PROMPT;
    if (iterations - i <= max_block_parallelism) break;
  }
}

template<class T>void sum(T* device_array, const Counter length)
{
  mean(device_array, length, true);
}

template<class T>CU_KER void sumKernel(T* device_array_result,
  T* device_array_a, T* device_array_b, const Counter length,
  const bool diff)
{
  Counter index = threadIndex;
  if (index >= length) return;
  if (diff)
    device_array_result[index] = device_array_a[index] - device_array_b[index];
  else
    device_array_result[index] = device_array_a[index] + device_array_b[index];
}

template<class T>void sum(T* device_array_result, T* device_array_a,
  T* device_array_b, const Counter length, const bool diff = false)
{
  dim3 threads, blocks;
  configureGrid(blocks, threads, length);
  sumKernel << < blocks, threads >> >(device_array_result, device_array_a,
    device_array_b, length, diff);
  cudaDeviceSynchronize();
  CU_PROMPT;
}

template<class T>CU_KER void scaleKernel(T* device_array_result,
  T* device_array, const real scale, const Counter length)
{
  Counter index = threadIndex;
  if (index >= length) return;
  device_array_result[index] = device_array[index] * scale;
}

template<class T>void scale(T* device_array_result, T* device_array,
  const real scale, const Counter length)
{
  dim3 threads, blocks;
  configureGrid(blocks, threads, length);
  scaleKernel << < blocks, threads >> >(device_array_result, device_array,
    scale, length);
  cudaDeviceSynchronize();
  CU_PROMPT;
}

#endif