#ifndef LIGHT_H
#define LIGHT_H

#include <Common/RayTracingStruct.h>
#include <Entities/RayTracingEntity.h>

/*!
@class Base class for lights, implementing a point light.
*/
class Light : virtual public LightStruct, public RayTracingEntity
{
public:

  Light(RayTracingEntityType type);

  RayTracingEntity* createCopy()const;

  RayTracingEntityId getIdentity()const;

  void update();
};

#endif
