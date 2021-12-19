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
  MaterialTypeMax,
  MaterialTypeReflection    = 0x1,
  MaterialTypeTransmission  = 0x2,
  MaterialTypeDiffuse       = 0x4,
  MaterialTypeGlossy        = 0x8,
  MaterialTypeSpecular      = 0x10,
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
  mat.flags = (mat.flags & 0xFFFFFF00) | (type & 0xFF);
}

static void setMaterialShader(MaterialStruct& mat, MaterialShaders shaderType)
{
  mat.flags = mat.flags | (shaderType << 8);
}

static void setMaterialSpecularExponent(MaterialStruct& mat, float value)
{
  mat.parameters.x = value;
}

static void setMaterialRefractiveIndex(MaterialStruct& mat, float value)
{
  mat.parameters.x = value;
}

static void setMaterialFresnelK(MaterialStruct& mat, float value)
{
  mat.parameters.y = value;
}

#else

inline ushort getMaterialType(const MaterialStruct mat)
{
  return mat.flags & 0xFF;
}

inline ushort getMaterialShader(const MaterialStruct mat)
{
  return mat.flags >> 8;
}

inline colorType getMaterialSpecularExponent(const MaterialStruct mat)
{
  return mat.parameters.x;
}

inline colorType getMaterialRefractiveIndex(const MaterialStruct mat)
{
  return mat.parameters.x;
}

inline colorType getMaterialFresnelK(const MaterialStruct mat)
{
  return mat.parameters.y;
}

inline colorType4 shadeMaterialAtIntersection(const MaterialStruct material, const float3 lightDirection, const float3 worldDirection, const HitStruct hit, const float3 hitNormal)
{
  colorType4 color = colorType4(0.f);
  const ushort shader = getMaterialShader(material);
#ifdef HitStructIndexIdentity
  // add diffuse color
  colorType diffuseScale;
  if (shader & MaterialShaderLambert)
  {
    diffuseScale = dot(lightDirection, hitNormal);
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
    const float3 reflection = reflectVector(lightDirection, hitNormal);
    specularScale = dot(reflection, worldDirection);
  }
  else
  if (shader & MaterialShaderBlinnPhong)
  {
    const float3 halfDir = normalize(lightDirection - worldDirection);
    specularScale = dot(halfDir, hitNormal);
  }
  else
  {
    specularScale = 0.f;
  }
  if (specularScale)
  {
    color += material.specular * pow(max(specularScale, (colorType)0.f), getMaterialSpecularExponent(material));
  }
#endif

  return color;
}

inline float fresnelDielectric(float cosThetaI, colorType etaI, colorType etaT)
{
  const float sinThetaI = sqrt(1.f - cosThetaI * cosThetaI);
  if (cosThetaI > 0.f)
  {
    const colorType temp = etaI;
    etaI = etaT;
    etaT = temp;
  }
  cosThetaI = abs(cosThetaI);

  const float sinThetaT = etaI / etaT * sinThetaI;
  if (sinThetaT >= 1.f)
  {
    return 1.f;
  }

  const float cosThetaT = sqrt(1.f - sinThetaT * sinThetaT);
  const float rPara = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
  const float rPerp = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));
  return (rPara * rPara + rPerp * rPerp) * 0.5f;
}

inline float fresnelConductor(float cosThetaI, colorType etaI, colorType etaT, const colorType k)
{
  return 0.f;
}

inline float getFresnelCoefficient(float cosThetaI, colorType etaI, colorType etaT, const colorType k)
{
  return select(fresnelDielectric(cosThetaI, etaI, etaT), fresnelConductor(cosThetaI, etaI, etaT, k), k);
}

#endif

#endif
