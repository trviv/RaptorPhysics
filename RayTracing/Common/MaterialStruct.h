#ifndef MATERIAL_STRUCT_H
#define MATERIAL_STRUCT_H

#define CompReflectance         components.y
#define CompTranslucency        components.z
#define CompInvRefractiveIndex  components.w


enum MaterialTypes
{
  MaterialTypePlastic,
  MaterialTypeMax
};

typedef struct IdentityInfo_t MaterialId;


/*!
@class Base class for Materials Data.
*/
struct MaterialStruct_t
{
  colorType4 diffuse;
  colorType4 specular;
  colorType4 emissive;
  colorType4 components;
};

typedef struct MaterialStruct_t MaterialStruct;


#ifndef COMPUTE_SHADER_SCOPE

static void setMaterialType(MaterialStruct& mat, MaterialTypes type)
{
  mat.components.x = type;
}

#else

inline ushort getMaterialType(const MaterialStruct mat)
{
  return asUshort(mat.components.x);
}

inline colorType4 shadeMaterialAtIntersection(const MaterialStruct material, const float3 lightDirection, const HitStruct hit)
{
  colorType4 color = material.emissive;
#ifdef HitStructNormal
  color += material.diffuse * dot(lightDirection, hit.normal);
#endif

  return color;
}

#endif

#endif
