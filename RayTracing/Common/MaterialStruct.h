#ifndef MATERIAL_STRUCT_H
#define MATERIAL_STRUCT_H

#define CompReflectance         components.y
#define CompTranslucency        components.z
#define CompInvRefractiveIndex  components.w


enum MaterialTypes
{
  MaterialTypePlastic,
  MaterialTypeReflective,
  MaterialTypeTranslucent,
  MaterialTypeMax
};

enum MaterialShaders
{
  MaterialShaderLambert     = 0x1,
  MaterialShaderPhong       = 0x2,
  MaterialShaderBlinnPhong  = 0x4,
};

typedef struct IdentityInfo_t MaterialId;


/*!
@class Base class for Materials Data.
*/
typedef struct DEFAULT_ALIGN
{
  colorType4 diffuse;
  colorType4 specular;
  colorType4 emissive;
  colorType4 parameters;
  uint flags;
  uint padding[3];
} MaterialStruct;


#ifndef COMPUTE_SHADER_SCOPE

static void setMaterialType(MaterialStruct& mat, MaterialTypes type)
{
  mat.flags = (mat.flags & 0xFFFFFFF0) | (type & 0xF);
}

static void setMaterialShader(MaterialStruct& mat, MaterialShaders shaderType)
{
  mat.flags = mat.flags | (shaderType << 4);
}

static void setMaterialSpecularExponent(MaterialStruct& mat, float value)
{
  mat.parameters.x = value;
}

static void setMaterialRefractiveIndex(MaterialStruct& mat, float value)
{
  mat.parameters.x = value;
}

#else

inline ushort getMaterialType(const MaterialStruct mat)
{
  return mat.flags & 0xF;
}

inline ushort getMaterialShader(const MaterialStruct mat)
{
  return mat.flags >> 4;
}

inline colorType getMaterialSpecularExponent(const MaterialStruct mat)
{
  return mat.parameters.x;
}

inline colorType getMaterialRefractiveIndex(const MaterialStruct mat)
{
  return mat.parameters.x;
}

inline colorType4 shadeMaterialAtIntersection(const MaterialStruct material, const float3 lightDirection, const float3 worldDirection, const HitStruct hit)
{
  colorType4 color = colorType4(0.f);
  const ushort shader = getMaterialShader(material);
#ifdef HitStructNormal
  // add diffuse color
  colorType diffuseScale;
  if (shader & MaterialShaderLambert)
  {
    diffuseScale = dot(lightDirection, hit.normal);
  }
  else
  {
    diffuseScale = 0.f;
  }
  color += material.diffuse * diffuseScale;

  // add specular color
  colorType specularScale;
  if (shader & MaterialShaderPhong)
  {
    const float3 reflection = reflectVector(lightDirection, hit.normal);
    specularScale = dot(reflection, worldDirection);
  }
  else
  if (shader & MaterialShaderBlinnPhong)
  {
    const float3 halfDir = normalize(lightDirection - worldDirection);
    specularScale = dot(halfDir, hit.normal);
  }
  else
  {
    specularScale = 0.f;
  }
  color += material.specular * pow(max(specularScale, (colorType)0.f), getMaterialSpecularExponent(material));
#endif

  return color;
}

#endif

#endif
