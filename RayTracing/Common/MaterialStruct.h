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


/*!
@struct Structor to uniquely represent a material.
*/
struct ALIGN(4) MaterialId_t
{
  uint identity;
};

typedef struct MaterialId_t MaterialId;


/*!
@class Base class for Materials Data.
*/
struct MaterialStruct_t
{
  half4 diffuse;
  half4 specular;
  half4 emissive;
  half4 components;
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

#endif

#endif
