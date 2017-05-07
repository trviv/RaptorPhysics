#ifndef __CONSTRAIN
#define __CONSTRAIN

#include "CudaUtils.h"

static bool constrain_debug = true;

#define defineGet(ret_type, name, ptr) \
CU_DEV_HOST ret_type* name()const{return ptr;} \
CU_DEV_HOST ret_type* name(){return ptr;}

/// function to offset an array
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

enum ObjectType
{
  OBJ_SYSTEM,
  OBJ_EQUATION,
  OBJ_CLOTH,
  OBJ_RIGID_BODY
};

enum ObjectArrays
{
  OBJ_CONSTRAINTS_ARRAY,
  OBJ_COEFFICIENTS_ARRAY,
  OBJ_CONSTRAINTS,
  OBJ_VALUES,
  OBJ_MASS,
  OBJ_DISTANCE,
  OBJ_POSITION,
  OBJ_PREDICTED_POS,
  OBJ_VELOCITY,
  OBJ_FORCE,
  OBJ_COM_OFFSET,
  OBJ_MATRIX,
  OBJ_DELTA_POS,
  OBJ_MAX_ARRAYS
};

class ObjectData
{
public:

  Counter offset[OBJ_MAX_ARRAYS];
  Counter count[OBJ_MAX_ARRAYS];

  Counter iterations = 32;
  real    particle_radius;
  ObjectType  type;

public:

  ObjectData();

  CU_DEV_HOST Counter getIterations()const
  {
    return iterations;
  }

  CU_DEV_HOST real getRadius()const
  {
    return particle_radius;
  }

  template<class IndexType, class CoefType, class ValueType> friend class ConstrainSolver;
};

/// Simple linear equality solver
template<class IndexType, class CoefType, class ValueType>
class ConstrainSolver :
  public DeviceEntity < ConstrainSolver<IndexType, CoefType, ValueType> >,
  public ObjectData
{
protected:

  // datatype for host coefficient
  typedef std::vector<CoefType>   SingleCoefficient;
  // datatype for host constrain
  typedef std::vector<IndexType>  SingleConstrain;

  // Device members

  IndexType*  device_constrain_array = NULL;
  CoefType*   device_coef_array = NULL;
  Constrain*  device_constrain = NULL;
  ValueType*  device_value_arrays = NULL;

  real*       device_mass = NULL;
  real*       device_distance = NULL;
  ValueType*  device_position = NULL;
  ValueType*  device_predicted_pos = NULL;
  ValueType*  device_velocity = NULL;
  ValueType*  device_force = NULL;
  ValueType*  device_com_offset = NULL;
  real*       device_matrix = NULL;
  ValueType*  device_del_pos = NULL;

  // Host members

  // 2d array containing coefficients for all the constraints
  std::vector <SingleCoefficient> constrain_coef;
  // value of constraints
  std::vector <ValueType>         constrain_values;
  // 2d array containing constrain map
  std::vector <SingleConstrain>   constraints;

  // point values
  std::vector<real>               point_mass;
  std::vector<std::vector<real>>  point_distance;

  // rigid body values
  std::vector<ValueType>    com_offset;

  std::vector<ObjectData>   objects;
  DeviceAllocator <__int8>  device_memory;

  Counter node_count = 0;

  // device allocator for thei instance
  ConstrainSolver<IndexType, CoefType, ValueType> *constrain_alloc = NULL;

  FORCE_INLINE void free()
  {
    constraints.clear();
    constrain_values.clear();
    constrain_coef.clear();
    point_mass.clear();
    point_distance.clear();
    com_offset.clear();
    objects.clear();

    constrain_alloc = NULL;

    device_constrain_array = NULL;
    device_coef_array = NULL;
    device_constrain = NULL;
    device_value_arrays = NULL;

    device_mass = NULL;
    device_distance = NULL;
    device_position = NULL;
    device_predicted_pos = NULL;
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
    type = OBJ_SYSTEM;
  }

  FORCE_INLINE ~ConstrainSolver()
  {
    free();
  }

  real  del_t = real(.1);
  real  velocity_fraction = real(.999);
  real  successiveOverRealaxation = real(1.5);

  void pushObject(ObjectType type);

  void addConstrain(const IndexType index, const IndexType connection);
  void addCoefficient(const IndexType index, const CoefType coef);
  void addCC(const IndexType index, const IndexType connection, const CoefType coef);
  void addDistance(const IndexType index, const IndexType connection, const real distance);
  void setValue(const IndexType index, const ValueType& value);
  void setInvMass(const IndexType index, const real inv_mass = 0);

  void show()const;

  void exportToDevice(__int8** device_additional_memory = NULL,
    Counter additional_size = 0, Counter baseSize = 0);

  void integrate();
  void differentiate();
  void solve();

  template<class T> void flatArray(std::vector<T>& out, const std::vector<std::vector<T>>& in)const;

  CU_HOST void configureGrid(dim3& blocks, dim3& threads, const Counter length)const
  {
    ::configureGrid(blocks, threads, length);
  }

  CU_HOST void configureGrid(dim3& blocks, dim3& threads)const
  {
    ::configureGrid(blocks, threads, getNodeCount());
  }

  CU_HOST std::vector<ValueType>& hostPosition()
  {
    return constrain_values;
  }

  CU_HOST std::vector<ValueType>& hostComOffset()
  {
    return com_offset;
  }

  CU_DEV_HOST Counter getNodeCount()const
  {
    return node_count;
  }

  defineGet(real, getMass, device_mass);
  defineGet(real, getDistance, device_distance);
  defineGet(ValueType, getPosition, device_position);
  defineGet(ValueType, getDelPosition, device_del_pos);
  defineGet(ValueType, getVelocity, device_velocity);
  defineGet(ValueType, getForce, device_force);
  defineGet(real, getMatrix, device_matrix);

  CU_DEV Constrain getConstrain(const IndexType index)const
  {
    return device_constrain[index];
  }

  CU_DEV CoefType getCoef(const IndexType index)const
  {
    return device_coef_array[index];
  }

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
};

#endif