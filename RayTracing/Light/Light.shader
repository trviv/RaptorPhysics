/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#ifndef LIGHT_SHADER
#define LIGHT_SHADER

inline bool sampleLight(Thread float3* color, Thread float3* direction, Thread float* maxDistance,
  Const LightStruct* light, const uint offset, const ushort bounce, const float3 hitPosition)
{
  if (getRayTracingEntityType(light->identity) == 2)
  {
    const float2 random = constructFloat2(getRandomNumber(offset, 2+bounce*4+0), getRandomNumber(offset, 2+bounce*4+1)) * 2.0f - 1.0f;
    //const half2 random = half2(getRandomNumber(offset, 2+bounce*4+0), getRandomNumber(offset, 2+bounce*4+1)) * 2.0f - 1.0f;
    const float3 position = light->position + constructFloat3(light->right.xyz) * random.x + constructFloat3(light->up.xyz) * random.y;
    *direction = position - hitPosition;
    *maxDistance = length(*direction);
    *direction /= *maxDistance;
    const float dotDirLight = -dot(*direction, light->normal);
    *color = light->color * dotDirLight;
    return dotDirLight >= MIN_TIME;
  }
  else
  {
    const float3 position = light->position;
    *color = light->color;
    *direction = position - hitPosition;
    *maxDistance = length(*direction);
    *direction /= *maxDistance;
  }

  return true;
}

#endif
