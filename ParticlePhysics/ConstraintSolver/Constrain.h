#ifndef __CONSTRAIN
#define __CONSTRAIN

#include "CudaUtils.h"

#define threadIndex threadIdx.x + threadIdx.y*blockDim.x + (blockIdx.x + blockIdx.y*gridDim.x)*blockDim.x*blockDim.y

static bool constrain_debug = true;

#define defineGet(ret_type, name, ptr) \
CU_DEV_HOST ret_type* name()const{return ptr;} \
CU_DEV_HOST ret_type* name(){return ptr;}

template<class DEST_T, class SRC_T> DEST_T* getPointer(SRC_T* source_ptr,
  const Counter& offset)
{
  return (DEST_T*)((__int8*)source_ptr + sizeof(SRC_T)*offset);
}

/// Class representing a constrain offset in an array and number of constrains
class Constrain
{
#define OFFSET_BITS 24
#define OFFSET_MASK 0xFFFFFF

protected:
  unsigned __int32 value;

public:
  CU_DEV_HOST Constrain(const __int32 offset = 0, const __int32 count = 0)
  {
    setOffset(offset);  setCount(count);
  }

  CU_DEV_HOST __int32 count()const
  {
    return value >> OFFSET_BITS;
  }

  CU_DEV_HOST __int32 offset()const
  {
    return value&OFFSET_MASK;
  }

  CU_DEV_HOST void setOffset(const __int32 offset)
  {
    value = ((value&(-1 ^ OFFSET_MASK)) | (offset&OFFSET_MASK));
  }

  CU_DEV_HOST void setCount(const __int32 count)
  {
    value = ((value&OFFSET_MASK) | (count << OFFSET_BITS));
  }
};

/// 
enum ConstrainBuffer
{
  DEF = 0,
  VAR0 = 1,
  VAR1 = 2
};

/// Simple linear equality solver
template<class IndexType, class CoefType, class ValueType>
class ConstrainSolver : public
  DeviceEntity < ConstrainSolver<IndexType, CoefType, ValueType> >
{
protected:

  // datatype for host coefficient
  typedef std::vector<CoefType>  SingleCoefficient;
  // datatype for host constrain
  typedef std::vector<IndexType>  SingleConstrain;

  // 2d array containing coefficients for all the constraints
  std::vector <SingleCoefficient> constrain_coef;
  // value of constraints
  std::vector <ValueType>         constrain_values;
  // 2d array containing constrain map
  std::vector <SingleConstrain>   constraints;

  // shared parameters
  IndexType   countraints_count = 0;
  IndexType   iterations = 32;
  IndexType   offset = 0;

  // device allocator for thei instance
  ConstrainSolver<IndexType, CoefType, ValueType> *constrain_alloc = NULL;

  //device array address
  IndexType   *device_constrain_array = NULL;
  CoefType    *device_coef_array = NULL;
  Constrain   *device_constrain = NULL;
  ValueType   *device_value_arrays = NULL;

  // point contraint
  std::vector<real>               point_mass;
  std::vector<std::vector<real>>  point_distance;

  real*       device_mass = NULL;
  real*       device_distance = NULL;
  ValueType*  device_position = NULL;
  ValueType*  device_velocity = NULL;
  ValueType*  device_force = NULL;
  ValueType*  device_com_offset = NULL;
  real*       device_matrix = NULL;
  ValueType*  device_del_pos = NULL;

  DeviceAllocator < __int8 >  device_memory;

  FORCE_INLINE void free()
  {
    constraints.clear();
    constrain_values.clear();
    constrain_coef.clear();

    constrain_alloc = NULL;

    device_constrain_array = NULL;
    device_coef_array = NULL;
    device_constrain = NULL;
    device_value_arrays = NULL;

    device_mass = NULL;
    device_distance = NULL;
    device_position = NULL;
    device_velocity = NULL;
    device_force = NULL;
    device_com_offset = NULL;
    device_matrix = NULL;
    device_del_pos = NULL;

    device_memory.free();
  }

  template<class type> FORCE_INLINE void expand(const IndexType index,
    std::vector<type>& list)
  {
    while (list.size() <= index)  list.push_back(type());
  }

public:

  FORCE_INLINE ConstrainSolver()
  {
  }

  FORCE_INLINE ~ConstrainSolver()
  {
    free();
  }

  real  del_t = real(.01);
  real  velocity_fraction = real(.999);
  real  successiveOverRealaxation = real(1.5);

  FORCE_INLINE void pushPreviousObject()
  {
    offset += constraints.size();
  }

  CU_DEV Constrain getConstrain(const IndexType index)const
  {
    return device_constrain[index];
  }

  CU_DEV CoefType getCoef(const IndexType index)const
  {
    return device_coef_array[index];
  }

  CU_DEV_HOST IndexType getIterations()const
  {
    return iterations;
  }

  CU_DEV_HOST IndexType getNodeCount()const
  {
    return countraints_count;
  }

  defineGet(real, getMass, device_mass);

  defineGet(real, getDistance, device_distance);

  defineGet(ValueType, getPosition, device_position);

  defineGet(ValueType, getDelPosition, device_del_pos);

  defineGet(ValueType, getVelocity, device_velocity);

  defineGet(ValueType, getForce, device_force);

  defineGet(real, getMatrix, device_matrix);

  CU_DEV const Real3* getComOffset()const
  {
    return device_com_offset;
  }

  CU_DEV_HOST const ValueType* getValueBuffer(const ConstrainBuffer buffer_index)const
  {
    return (device_value_arrays + buffer_index*getNodeCount());
  }

  CU_DEV_HOST ValueType* getValueBuffer(const ConstrainBuffer buffer_index)
  {
    return (device_value_arrays + buffer_index*getNodeCount());
  }

  CU_DEV const ValueType& getValue(const IndexType index,
    const ConstrainBuffer buffer_index)const
  {
    return device_value_arrays[buffer_index*getNodeCount() + index];
  }

  CU_DEV ValueType& getValue(const IndexType index,
    const ConstrainBuffer buffer_index)
  {
    return device_value_arrays[buffer_index*getNodeCount() + index];
  }

  CU_DEV ValueType& getValue(const IndexType index)
  {
    return device_value_arrays[index];
  }

  CU_DEV IndexType getIndex(const IndexType offset)const
  {
    return device_constrain_array[offset];
  }

  FORCE_INLINE void add(const IndexType index, const IndexType connection,
    const CoefType coef)
  {
    // add contrain and coeficient to first element
    expand<SingleConstrain>(index + offset, constraints);
    constraints[index + offset].push_back(connection + offset);
    expand<SingleCoefficient>(index + offset, constrain_coef);
    constrain_coef[index + offset].push_back(coef);
  }

  FORCE_INLINE void addValue(const IndexType index, const ValueType& value)
  {
    //add value for the element
    expand<ValueType>(index + offset, constrain_values);
    constrain_values[index + offset] = value;
  }

  FORCE_INLINE void show()const
  {
    for (Counter i = 0; i < constraints.size(); i++)
    {
      std::cout << " Element " << i << " is linked with: ";
      for (Counter j = 0; j < constraints[i].size(); j++)
      {
        std::cout << constraints[i][j] << " ";
      }
      std::cout << "\n";
    }
  }

  void exportToDevice(__int8** device_additional_memory = NULL,
    Counter additional_size = 0, Counter baseSize = 0)
  {
    if (constrain_alloc) return;
    if (!baseSize) baseSize = sizeof(*this);
    // arrays to be exported to device
    std::vector <IndexType>   constrain_array;
    std::vector <CoefType>    constrain_coef;
    std::vector <Constrain>   constrain_header;

    // create flat constrain array for device
    for (Counter i = 0; i < constraints.size(); i++)
    {
      constrain_header.push_back(Constrain(constrain_array.size(), 0));
      for (Counter j = 0; j < constraints[i].size(); j++)
      {
        constrain_array.push_back(constraints[i][j]);
        constrain_coef.push_back(this->constrain_coef[i][j]);
      }
      constrain_header[i].setCount(constraints[i].size());  // the constrain header
    }

    // total size for allocation
    IndexType total_allocation = baseSize +
      constrain_array.size()*sizeof(IndexType) +
      constrain_coef.size()*sizeof(CoefType) +
      constrain_header.size()*sizeof(Constrain) +
      constrain_values.size()*sizeof(ValueType) * 3 +
      additional_size;

    device_memory.alloc(total_allocation);
    countraints_count = constraints.size();

    constrain_alloc = (ConstrainSolver<IndexType, CoefType, ValueType>*)device_memory.get();
    device_constrain_array = getPointer < IndexType,
      ConstrainSolver < IndexType, CoefType, ValueType >> (constrain_alloc, 1);
    //device_constrain_array = (IndexType*)((__int8*)constrain_alloc + baseSize);
    device_coef_array = getPointer<CoefType, IndexType>(device_constrain_array,
      constrain_array.size());
    //device_coef_array = (CoefType*)((__int8*)device_constrain_array +
    //sizeof(IndexType)*constrain_array.size());
    device_constrain = getPointer<Constrain, CoefType>(device_coef_array,
      constrain_coef.size());
    //device_constrain = (Constrain*)((__int8*)device_coef_array +
    //sizeof(CoefType)*constrain_coef.size());
    device_value_arrays = getPointer<ValueType, Constrain>(device_constrain,
      constrain_header.size());
    //device_value_arrays = (ValueType*)((__int8*)device_constrain +
    //sizeof(Constrain)*constrain_header.size());
    if (additional_size)
    {
      *device_additional_memory = getPointer<__int8, ValueType>(
        device_value_arrays, constrain_values.size() * 3);
      //*device_additional_memory = ((__int8*)device_value_arrays +
      //constrain_values.size() * 3 * sizeof(ValueType));
    }

    std::vector <ValueType> value_array = constrain_values;
    for (Counter i = 0; i < 2 * countraints_count; i++) value_array.push_back(0);

    DeviceEntity::exportToDevice(this, constrain_alloc, 1);
    CU_PROMPT;
    DeviceEntity<IndexType>::exportToDevice(&constrain_array[0],
      device_constrain_array, constrain_array.size());
    CU_PROMPT;
    DeviceEntity<CoefType>::exportToDevice(&constrain_coef[0],
      device_coef_array, constrain_coef.size());
    CU_PROMPT;
    DeviceEntity<Constrain>::exportToDevice(&constrain_header[0],
      device_constrain, constrain_header.size());
    CU_PROMPT;
    DeviceEntity<ValueType>::exportToDevice(&value_array[0],
      device_value_arrays, value_array.size());
    CU_PROMPT;

    std::cout << "Constrain Alloc:\n";
    std::cout << "Base: " << baseSize <<
      "\tAddr: " << (__int64)constrain_alloc << "\n";
    std::cout << "Constrain array: " <<
      constrain_array.size()*sizeof(IndexType) <<
      "\tAddr: " << (__int64)device_constrain_array <<
      "\tDiff: " << (__int64)device_constrain_array - (__int64)constrain_alloc << "\n";
    std::cout << "Constrain coef: " <<
      constrain_coef.size()*sizeof(CoefType) <<
      "\tAddr: " << (__int64)device_coef_array <<
      "\tDiff: " << (__int64)device_coef_array - (__int64)device_constrain_array << "\n";
    std::cout << "Constrain header: " <<
      constrain_header.size()*sizeof(Constrain) <<
      "\tAddr: " << (__int64)device_constrain <<
      "\tDiff: " << (__int64)device_constrain - (__int64)device_coef_array << "\n";
    std::cout << "Constrain values: " <<
      constrain_values.size()*sizeof(ValueType) * 3 <<
      "\tAddr: " << (__int64)device_value_arrays <<
      "\tDiff: " << (__int64)device_value_arrays - (__int64)device_constrain << "\n";
    std::cout << "Additional: " << additional_size <<
      "\tAddr: " << (__int64)(*device_additional_memory) <<
      "\tDiff: " << (__int64)(*device_additional_memory) - (__int64)device_value_arrays << "\n";
  }

  void solve();

  CU_HOST void configureGrid(dim3& blocks, dim3& threads, const Counter length)const
  {
    ::configureGrid(blocks, threads, length);
  }

  CU_HOST void configureGrid(dim3& blocks, dim3& threads)const
  {
    ::configureGrid(blocks, threads, getNodeCount());
  }

  void integrate();
  void differentiate();
};

template<class IndexType, class CoefType, class ValueType>
CU_KER void constrainSolver(
  ConstrainSolver<IndexType, CoefType, ValueType>* constrain, Counter iteration)
{
  // the weight for new value
  const CoefType weight = CoefType(3. / 4.);// 1. / 2.;

  // kernel grid size
  //const IndexType grid = blockDim.x*blockDim.y;
  const IndexType values_count = constrain->getNodeCount();

  //for (IndexType i = 0; i < constrain->getIterations(); i++)
  {
    const ConstrainBuffer old_value = (iteration & 1) == 0 ? VAR0 : VAR1;
    const ConstrainBuffer new_value = (old_value == VAR0 ? VAR1 : VAR0);

    //IndexType iteration = ceil(values_count / float(grid));

    //for (iteration = iteration - 1; iteration >= 0; iteration--)
    {
      const IndexType index = threadIndex;// threadIdx.x + threadIdx.y*blockDim.x + iteration*grid;
      if (index < values_count)
      {
        ValueType sum = 0;
        IndexType count = constrain->getConstrain(index).count();
        IndexType offset = constrain->getConstrain(index).offset();
        offset++;
        for (count = count - 2; count >= 0; count--)
        {
          sum += constrain->getCoef(offset) *
            constrain->getValue(constrain->getIndex(offset), old_value);
          offset++;
        }
        constrain->getValue(index, new_value) =
          (1. - weight)*constrain->getValue(index, old_value) +
          weight*(constrain->getValue(index) - sum) /
          constrain->getCoef(constrain->getConstrain(index).offset());
      }
    }
  }
}

template<class IndexType, class CoefType, class ValueType>
void ConstrainSolver<IndexType, CoefType, ValueType>::solve()
{
  exportToDevice();
  dim3 threads, blocks;
  configureGrid(blocks, threads);

  for (Counter i = 0; i < iterations; i++)
  {
    constrainSolver<IndexType, CoefType, ValueType>
      << <blocks, threads >> >(constrain_alloc, i);
    cudaDeviceSynchronize();
    CU_PROMPT;
  }

  ValueType* values = new ValueType[getNodeCount()];
  DeviceEntity<ValueType>::importToHost(values,
    &(device_value_arrays[((iterations & 1) + 1)*getNodeCount()]),
    getNodeCount());

  for (Counter i = 0; i < getNodeCount(); i++)
    std::cout << values[i] << "\n";

  delete values;
}

template<class IndexType, class CoefType, class ValueType>
CU_KER void constrainIntegrate(
  ConstrainSolver<IndexType, CoefType, ValueType>* constrain)
{
  const IndexType index = threadIndex;
  if (index < constrain->getNodeCount())
  {
    /*
    constrain->getVelocity()[index] += (constrain->del_t*
    constrain->velocity_fraction*constrain->getMass()[index])*
    constrain->getForce()[index];
    */
    constrain->getPosition()[index] += constrain->getVelocity()[index] *
      constrain->del_t;
  }
}

template<class IndexType, class CoefType, class ValueType>
CU_KER void constrainDifferentiate(
  ConstrainSolver<IndexType, CoefType, ValueType>* constrain)
{
  const IndexType index = threadIndex;
  if (index < constrain->getNodeCount())
  {
    constrain->getVelocity()[index] = constrain->getDelPosition()[index] /
      constrain->del_t;
    constrain->getForce()[index] = constrain->getMass()[index] *
      constrain->getVelocity()[index] /
      constrain->del_t;
  }
}

template<class IndexType, class CoefType, class ValueType>
void ConstrainSolver<IndexType, CoefType, ValueType>::integrate()
{
  //DeviceEntity<ValueType>::copy(getValueBuffer(VAR1), getValueBuffer(DEF), getNodeCount());
  CU_PROMPT;
  dim3 threads, blocks;
  configureGrid(blocks, threads);
  constrainIntegrate << <blocks, threads >> >(constrain_alloc);
  cudaDeviceSynchronize();
  CU_PROMPT
}

template<class IndexType, class CoefType, class ValueType>
void ConstrainSolver<IndexType, CoefType, ValueType>::differentiate()
{
  dim3 threads, blocks;
  configureGrid(blocks, threads);
  constrainDifferentiate << <blocks, threads >> >(constrain_alloc);
  cudaDeviceSynchronize();
  CU_PROMPT
}

#endif