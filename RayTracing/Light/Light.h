/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

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

  void setIdentity(RayTracingEntityId identity);

  void update();
};

#endif
