#ifndef COLLISION_SOLVER_SHARED_H
#define COLLISION_SOLVER_SHARED_H

#ifndef COMPUTE_SHADER_SCOPE

#include "../../Common/ParticleStruct.h"
#include "../SharedAllocator.h"

#endif

#define COLLISION_COMPONENT_MORTON_CODE_MASK 1023

struct BVHLeafInfo_t
{
  uint mortonCode;
  uint index;
};

typedef struct BVHLeafInfo_t BVHLeafInfo;


struct BVHNodeInfo_t
{
  uint node[2];
};

typedef struct BVHNodeInfo_t BVHNodeInfo;

struct XAB_t
{
  float3 min;
  float3 max;
};

typedef struct XAB_t XAB;

#define addXAB(a, b)    { (a)->min = min((a)->min, (b)->min); (a)->max = max((a)->max, (b)->max);}
#define divXAB(a, b)    { (a)->min /= (*b); (a)->max /= (*b);}
#define clearXAB(a, b)  { }

static uint arrange32Bits(uint x)
{
  //........ ........ ......12 3456789A  //x
  //....1..2 ..3..4.. 5..6..7. .8..9..A  //x after interleaving bits

  //......12 3456789A ......12 3456789A  //x ^ (x << 16)
  //11111111 ........ ........ 11111111  //0x FF 00 00 FF
  //......12 ........ ........ 3456789A  //x = (x ^ (x << 16)) & 0xFF0000FF;

  //......12 ........ 3456789A 3456789A  //x ^ (x <<  8)
  //......11 ........ 1111.... ....1111  //0x 03 00 F0 0F
  //......12 ........ 3456.... ....789A  //x = (x ^ (x <<  8)) & 0x0300F00F;

  //..12..12 ....3456 3456.... 789A789A  //x ^ (x <<  4)
  //......11 ....11.. ..11.... 11....11  //0x 03 0C 30 C3
  //......12 ....34.. ..56.... 78....9A  //x = (x ^ (x <<  4)) & 0x030C30C3;

  //....1212 ..3434.. 5656..78 78..9A9A  //x ^ (x <<  2)
  //....1..1 ..1..1.. 1..1..1. .1..1..1  //0x 09 24 92 49
  //....1..2 ..3..4.. 5..6..7. .8..9..A  //x = (x ^ (x <<  2)) & 0x09249249;

  //........ ........ ......11 11111111  //0x000003FF

  x = (x ^ (x << 16)) & 0xFF0000FF;
  x = (x ^ (x << 8)) & 0x0300F00F;
  x = (x ^ (x << 4)) & 0x030C30C3;
  x = (x ^ (x << 2)) & 0x09249249;

  return x;
}

#ifdef COMPUTE_SHADER_SCOPE

uint get32BitMortonCode(const Device ParticleStruct* particle)
{
  uint x = ((uint)particle->position.x) & COLLISION_COMPONENT_MORTON_CODE_MASK;
  uint y = ((uint)particle->position.y) & COLLISION_COMPONENT_MORTON_CODE_MASK;
  uint z = ((uint)particle->position.z) & COLLISION_COMPONENT_MORTON_CODE_MASK;

  return arrange32Bits(x) | (arrange32Bits(y) << 1) | (arrange32Bits(z) << 2);
}

#endif

#endif