#ifndef LIGHT_H
#define LIGHT_H

#include <Common/RayTracingStruct.h>
#include <Entities/RayTracingEntity.h>

/*!
@class Base class represeting a light, implementing a point light.
*/
class Light : public LightStruct, public RayTracingEntity
{
public:

  Light();

  virtual void update()=0;
};

#endif
