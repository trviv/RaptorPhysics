#include "Constrain.h"

template<class IndexType, class CoefType, class ValueType>
template<class T> void ConstrainSolver<IndexType, CoefType, ValueType>::flatArray(
  std::vector<T>& out, const std::vector<std::vector<T>>& in)const
{
  out.clear();
  for (Counter i = 0; i < in.size(); i++)
    for (Counter j = 0; j < in[i].size(); j++)
      out.push_back(in[i][j]);
}

ObjectData defaultObj;

ObjectData::ObjectData()
{
  for (Counter i = 0; i < OBJ_MAX_ARRAYS; i++)
  {
    offset[i] = 0;
    count[i] = 0;
  }
}

template<class IndexType, class CoefType, class ValueType>
void ConstrainSolver<IndexType, CoefType, ValueType>::pushObject(ObjectType type)
{
  ObjectData obj;

  obj.type = type;
  std::vector <IndexType> constrain_array;
  flatArray<IndexType>(constrain_array, constraints);

  std::vector <CoefType>  constrain_coef;
  flatArray<CoefType>(constrain_coef, this->constrain_coef);

  std::vector <CoefType>  flat_distance;
  flatArray<real>(flat_distance, point_distance);

  obj.count[OBJ_CONSTRAINTS_ARRAY] = constrain_array.size();
  obj.count[OBJ_COEFFICIENTS_ARRAY] = constrain_coef.size();
  obj.count[OBJ_CONSTRAINTS] = constraints.size();
  obj.count[OBJ_VALUES] = constraints.size() * 3;
  obj.count[OBJ_MASS] = point_mass.size();
  obj.count[OBJ_DISTANCE] = flat_distance.size();
  obj.count[OBJ_POSITION] = constraints.size();
  obj.count[OBJ_PREDICTED_POS] = constraints.size();
  obj.count[OBJ_VELOCITY] = constraints.size();
  obj.count[OBJ_FORCE] = constraints.size();
  obj.count[OBJ_COM_OFFSET] = com_offset.size();
  obj.count[OBJ_MATRIX] = constraints.size() * 9;
  obj.count[OBJ_DELTA_POS] = constraints.size();

  if (objects.size() != 0)
  {
    ObjectData prev = objects.back();
    for (Counter i = 0; i < OBJ_MAX_ARRAYS; i++)
    {
      obj.offset[i] = prev.offset[i] + prev.count[i];
      obj.count[i] -= prev.count[i];
    }
  }
  objects.push_back(obj);
  node_count = objects[objects.size() - 1].offset[OBJ_CONSTRAINTS] +
    objects[objects.size() - 1].count[OBJ_CONSTRAINTS];
  show();
}

template<class IndexType, class CoefType, class ValueType>
void ConstrainSolver<IndexType, CoefType, ValueType>::addConstrain(
  const IndexType index, const IndexType connection)
{
  // add contrain to first element
  expand<SingleConstrain>(index + getNodeCount(), constraints);
  constraints[index + getNodeCount()].push_back(connection + getNodeCount());
}

template<class IndexType, class CoefType, class ValueType>
void ConstrainSolver<IndexType, CoefType, ValueType>::addCoefficient(
  const IndexType index, const CoefType coef)
{
  // add coeficient to first element
  expand<SingleCoefficient>(index + getNodeCount(), constrain_coef);
  constrain_coef[index + getNodeCount()].push_back(coef);
}

template<class IndexType, class CoefType, class ValueType>
void ConstrainSolver<IndexType, CoefType, ValueType>::addCC(
  const IndexType index, const IndexType connection, const CoefType coef)
{
  addConstrain(index, connection);
  addCoefficient(index, coef);
}

template<class IndexType, class CoefType, class ValueType>
void ConstrainSolver<IndexType, CoefType, ValueType>::addDistance(
  const IndexType index, const IndexType connection, const real distance)
{
  addConstrain(index, connection);
  expand<std::vector<real>>(index + getNodeCount(), point_distance);
  point_distance[index + getNodeCount()].push_back(distance);
}

template<class IndexType, class CoefType, class ValueType>
void ConstrainSolver<IndexType, CoefType, ValueType>::setValue(
  const IndexType index, const ValueType& value)
{
  //add value for the element
  expand<ValueType>(index + getNodeCount(), constrain_values);
  constrain_values[index + getNodeCount()] = value;
}

template<class IndexType, class CoefType, class ValueType>
void ConstrainSolver<IndexType, CoefType, ValueType>::setInvMass(
  const IndexType index, const real inv_mass)
{
  expand<real>(index + getNodeCount(), point_mass);
  point_mass[index + getNodeCount()] = inv_mass;
}

template<class IndexType, class CoefType, class ValueType>
void ConstrainSolver<IndexType, CoefType, ValueType>::show()const
{
  Counter curr_offset, end;
  curr_offset = objects[objects.size() - 1].offset[OBJ_CONSTRAINTS];
  end = getNodeCount();

  for (Counter i = curr_offset; i < end; i++)
  {
    std::cout << " Element " << i << " is linked with: ";
    for (Counter j = 0; j < constraints[i].size(); j++)
      std::cout << constraints[i][j] << " ";
    std::cout << "\n";
  }
}

template<class IndexType, class CoefType, class ValueType>
void ConstrainSolver<IndexType, CoefType, ValueType>::exportToDevice(
  __int8** device_additional_memory, Counter additional_size, Counter baseSize)
{
  if (constrain_alloc) return;
  if (!baseSize) baseSize = sizeof(*this);

  // arrays to be exported to device
  std::vector <IndexType>   constrain_array;
  std::vector <CoefType>    constrain_coef;
  std::vector <Constrain>   constrain_header;
  std::vector<real>         flat_distance;

  // create flat constrain array for device
  for (Counter i = 0; i < constraints.size(); i++)
  {
    constrain_header.push_back(Constrain(constrain_array.size(), 0));
    for (Counter j = 0; j < constraints[i].size(); j++)
      constrain_array.push_back(constraints[i][j]);
    constrain_header[i].setCount(constraints[i].size());  // the constrain header
  }

  flatArray<CoefType>(constrain_coef, this->constrain_coef);
  flatArray<CoefType>(flat_distance, this->point_distance);

  // total size for allocation
  Counter total_allocation = baseSize;
  ObjectArrays value_arrays[] = { OBJ_VALUES,
    OBJ_POSITION,
    OBJ_PREDICTED_POS,
    OBJ_VELOCITY,
    OBJ_FORCE,
    OBJ_COM_OFFSET,
    OBJ_DELTA_POS };
  void** value_arrays_ptr[] = { (void**)&device_value_arrays,
    (void**)&device_position,
    (void**)&device_predicted_pos,
    (void**)&device_velocity,
    (void**)&device_force,
    (void**)&device_com_offset,
    (void**)&device_del_pos };

  ObjectArrays real_arrays[] = { OBJ_MASS,
    OBJ_DISTANCE,
    OBJ_MATRIX };
  void** real_arrays_ptr[] = { (void**)&device_mass,
    (void**)&device_distance,
    (void**)&device_matrix };

  for (Counter i = 0; i < objects.size(); i++)
  {
    const ObjectData* obj = &objects[i];
    total_allocation += sizeof(IndexType)*obj->count[OBJ_CONSTRAINTS_ARRAY];
    total_allocation += sizeof(CoefType)*obj->count[OBJ_COEFFICIENTS_ARRAY];
    total_allocation += sizeof(Constrain)*obj->count[OBJ_CONSTRAINTS];

    for (auto value_array : value_arrays)
      total_allocation += sizeof(ValueType)*obj->count[value_array];

    for (auto real_array : real_arrays)
      total_allocation += sizeof(real)*obj->count[real_array];
  }

  device_memory.alloc(total_allocation);
  constrain_alloc = (ConstrainSolver<IndexType, CoefType, ValueType>*)device_memory.get();

  __int8* arrays = (__int8*)constrain_alloc + baseSize;

  ObjectData *last_obj = &objects[objects.size() - 1];

  device_coef_array = (CoefType*)arrays;
  if (constrain_coef.size())
    DeviceEntity<CoefType>::exportToDevice(&constrain_coef[0],
    device_coef_array, constrain_coef.size());
  else
    device_coef_array = NULL;

  arrays += sizeof(CoefType)*(last_obj->offset[OBJ_COEFFICIENTS_ARRAY] + last_obj->count[OBJ_COEFFICIENTS_ARRAY]);
  std::cout << "\nIndex: " << OBJ_COEFFICIENTS_ARRAY <<
    "\tAddr: " << (__int64)device_coef_array <<
    "\tSize: " << sizeof(CoefType)*(last_obj->offset[OBJ_COEFFICIENTS_ARRAY] + last_obj->count[OBJ_COEFFICIENTS_ARRAY]) << "\n";

  device_constrain = (Constrain*)arrays;
  DeviceEntity<Constrain>::exportToDevice(&constrain_header[0],
    device_constrain, constrain_header.size());
  arrays += sizeof(Constrain)*(last_obj->offset[OBJ_CONSTRAINTS] + last_obj->count[OBJ_CONSTRAINTS]);
  std::cout << "\nIndex: " << OBJ_CONSTRAINTS <<
    "\tAddr: " << (__int64)device_constrain <<
    "\tSize: " << sizeof(Constrain)*(last_obj->offset[OBJ_CONSTRAINTS] + last_obj->count[OBJ_CONSTRAINTS]) << "\n";

  device_constrain_array = (IndexType*)arrays;
  DeviceEntity<IndexType>::exportToDevice(&constrain_array[0],
    device_constrain_array, constrain_array.size());
  arrays += sizeof(IndexType)*(last_obj->offset[OBJ_CONSTRAINTS_ARRAY] + last_obj->count[OBJ_CONSTRAINTS_ARRAY]);
  std::cout << "\nIndex: " << OBJ_CONSTRAINTS_ARRAY <<
    "\tAddr: " << (__int64)device_constrain_array <<
    "\tSize: " << sizeof(IndexType)*(last_obj->offset[OBJ_CONSTRAINTS_ARRAY] + last_obj->count[OBJ_CONSTRAINTS_ARRAY]) << "\n";

  for (Counter i = 0; i < sizeof(value_arrays) / sizeof(value_arrays[0]); i++)
  {
    *(value_arrays_ptr[i]) = (void*)arrays;
    arrays += sizeof(ValueType)*(last_obj->offset[value_arrays[i]] + last_obj->count[value_arrays[i]]);
    std::cout << "\nIndex: " << value_arrays[i] <<
      "\tAddr: " << (__int64)*(value_arrays_ptr[i]) <<
      "\tSize: " << sizeof(ValueType)*(last_obj->offset[value_arrays[i]] + last_obj->count[value_arrays[i]]) << "\n";
  }
  std::vector <ValueType> value_array = constrain_values;
  for (Counter i = 0; i < 2 * getNodeCount(); i++)
    value_array.push_back(0);
  DeviceEntity<ValueType>::exportToDevice(&value_array[0],
    device_value_arrays, value_array.size());

  for (Counter i = 0; i < sizeof(real_arrays) / sizeof(real_arrays[0]); i++)
  {
    *(real_arrays_ptr[i]) = (void*)arrays;
    arrays += sizeof(real)*(last_obj->offset[real_arrays[i]] + last_obj->count[real_arrays[i]]);
    std::cout << "\nIndex: " << real_arrays[i] <<
      "\tAddr: " << (__int64)*(real_arrays_ptr[i]) <<
      "\tSize: " << sizeof(real)*(last_obj->offset[real_arrays[i]] + last_obj->count[real_arrays[i]]) << "\n";
  }

  if (point_mass.size())
    DeviceEntity<real>::exportToDevice(&point_mass[0],
    device_mass, point_mass.size());
  else
    device_mass = NULL;

  if (flat_distance.size())
    DeviceEntity<real>::exportToDevice(&flat_distance[0],
    device_distance, flat_distance.size());
  else
    device_distance = NULL;

  DeviceEntity<ValueType>::exportToDevice(&constrain_values[0], device_position,
    getNodeCount());
  DeviceEntity<ValueType>::set(device_velocity, 0, getNodeCount());
  DeviceEntity<ValueType>::set(device_force, 0, getNodeCount());
  DeviceEntity<ValueType>::set(device_del_pos, 0, getNodeCount());

  if (com_offset.size())
    DeviceEntity<ValueType>::exportToDevice(&com_offset[0], device_com_offset,
    com_offset.size());
  else
    device_com_offset = NULL;

  DeviceEntity<real>::set(device_matrix, 0, 9 * getNodeCount());
  DeviceEntity<ConstrainSolver<IndexType, CoefType, ValueType>>::exportToDevice(this, constrain_alloc);
}

template<class IndexType, class CoefType, class ValueType>
CU_KER void constrainIntegrate(
  ConstrainSolver<IndexType, CoefType, ValueType>* constrain)
{
  const IndexType index = threadIndex;
  if (index >= constrain->getNodeCount()) return;

  constrain->getVelocity()[index] += (
    constrain->del_t* constrain->velocity_fraction * constrain->getMass()[index])*
    constrain->getForce()[index];
  constrain->getPosition()[index] += constrain->getVelocity()[index] *
    constrain->del_t;
  if (constrain->getPosition()[index][Y] <= -2)
    constrain->getPosition()[index][Y] = -2;
  constrain->getForce()[index] = Real3(0, -9.8, 0);
}

template<class IndexType, class CoefType, class ValueType>
CU_KER void constrainDifferentiate(
  ConstrainSolver<IndexType, CoefType, ValueType>* constrain)
{
  const IndexType index = threadIndex;
  if (index >= constrain->getNodeCount()) return;

  constrain->getVelocity()[index] = constrain->getDelPosition()[index] /
    constrain->del_t;
  //constrain->getForce()[index] += constrain->getMass()[index] *
  //constrain->getVelocity()[index] /
  //constrain->del_t;
}

template<class IndexType, class CoefType, class ValueType>
void ConstrainSolver<IndexType, CoefType, ValueType>::integrate()
{
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
      const IndexType index = threadIndex;
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

#define classPrefix(x, y, z) template void ConstrainSolver<x, y, z>

#define declareFunctions(x, y, z) \
  classPrefix(x, y, z)::pushObject(ObjectType type); \
  classPrefix(x, y, z)::flatArray<x>(std::vector<x>& out, const std::vector<std::vector<x>>& in)const; \
  classPrefix(x, y, z)::flatArray<y>(std::vector<y>& out, const std::vector<std::vector<y>>& in)const; \
  classPrefix(x, y, z)::addConstrain(const x index, const x connection); \
  classPrefix(x, y, z)::addCoefficient(const x index, const y coef); \
  classPrefix(x, y, z)::addCC(const x index, const x connection, const y coef); \
  classPrefix(x, y, z)::addDistance(const x index, const x connection, const real distance); \
  classPrefix(x, y, z)::setValue(const x index, const z& value); \
  classPrefix(x, y, z)::setInvMass(const x index, const real inv_mass); \
  classPrefix(x, y, z)::show()const; \
  classPrefix(x, y, z)::exportToDevice(__int8** device_additional_memory, Counter additional_size, Counter baseSize); \
  classPrefix(x, y, z)::solve();

declareFunctions(__int32, real, Real3)
declareFunctions(__int32, real, real)

classPrefix(__int32, real, Real3)::integrate();
classPrefix(__int32, real, Real3)::differentiate();