#ifndef ACCELARATION_DATA_STRUCT_H
#define ACCELARATION_DATA_STRUCT_H

#include <Common/RayTracingStruct.h>

/*!
@class Base class for all acceleration structures.
*/
class AccelerationDataStruct
{
protected:
  DeviceArray<XAB>    groupBoundingBoxes;
  DeviceArray<uint>   groupPrimitiveIndex;

public:

  virtual void update();

  virtual void processRays(ComputeMemory* hitInfo, ComputeMemory* rays);
};

#endif
