#ifndef MATERIAL_H
#define MATERIAL_H

#include <Common/RayTracingStruct.h>
#include <Common/MaterialStruct.h>
#include <Entities/RayTracingEntity.h>


/*!
@class Base class for Materials.
*/
class Material : public MaterialStruct
{
public:

  Material(MaterialTypes type);
};

#endif
