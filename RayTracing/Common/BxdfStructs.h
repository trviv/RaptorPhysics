/*
 * Copyright (c) Vivek Trivedi, 2025
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#ifndef BXDF_STRUCTS_H
#define BXDF_STRUCTS_H

#include "BxdfCommon.h"

#define MAX_BXDFS 2

struct BXDFStruct {
  union {
    int dummy1;
    BXDFType type;
  };

  union {
    int dummy2;
    TransportMode mode;
  };

  union {
    float3 scale;         // SpecularReflection
    float3 reflection;    // ScaledBXDF, LambertianReflection
    float3 transmission;  // SpecularTransmission, LambertianTransmission
    float3 fresnelRefl;   // FresnelSpecular
  };

  union {
    float3 fresnelTrans;  // FresnelSpecular
    struct {              // OrenNayar
      float A, B;
    };
  };

  Fresnel fresnel;
};

struct BXDFS {
  uint32_t nBxDFs;
  BXDFStruct bxdfs[MAX_BXDFS];
};

struct ShadingStruct {
  float3 ns, ng;
  float3 ss, ts;
};

float3 WorldToLocal(const float3 v, const ShadingData shading) const {
  return float3(dot(v, shading.ss), dot(v, shading.ts), dot(v, shading.ns));
}

float3 LocalToWorld(const float3 v, const ShadingData shading) const {
  return float3(shading.ss.x * v.x + shading.ts.x * v.y + shading.ns.x * v.z,
                shading.ss.y * v.x + shading.ts.y * v.y + shading.ns.y * v.z,
                shading.ss.z * v.x + shading.ts.z * v.y + shading.ns.z * v.z);
}

struct SampleStruct {
  float3 wi;
  float3 f;
  float pdf;
};

//// BXDF Declarations
//class BXDF {
//public:
//  // BXDF Interface
//  virtual ~BXDF() {}
//  BXDF(BXDFType type) : type(type) {}
//  bool MatchesFlags(BXDFType t) const { return (type & t) == type; }
//  virtual float3 f(const float3 wo, const float3 wi) const = 0;
//  virtual float3 Sample_f(const float3 wo, float3 *wi,
//                          const Point2f &sample, float *pdf,
//                          BXDFType *sampledType = nullptr) const;
//  virtual float3 rho(const float3 wo, int nSamples,
//                     const Point2f *samples) const;
//  virtual float3 rho(int nSamples, const Point2f *samples1,
//                     const Point2f *samples2) const;
//  virtual float Pdf(const float3 wo, const float3 wi) const;
//  virtual std::string ToString() const = 0;
//
//  // BXDF Public Data
//  const BXDFType type;
//};

//// BXDF Declarations
//struct BXDF {
//  // BXDF Interface
//  virtual ~BXDF() {}
//  BXDF(BXDFType type) : type(type) {}
//  bool MatchesFlags(BXDFType t) const { return (type & t) == type; }
//  virtual float3 f(const float3 wo, const float3 wi) const = 0;
//  virtual float3 Sample_f(const float3 wo, float3 *wi,
//                          const Point2f &sample, float *pdf,
//                          BXDFType *sampledType = nullptr) const;
//  virtual float3 rho(const float3 wo, int nSamples,
//                     const Point2f *samples) const;
//  virtual float3 rho(int nSamples, const Point2f *samples1,
//                     const Point2f *samples2) const;
//  virtual float Pdf(const float3 wo, const float3 wi) const;
//  virtual std::string ToString() const = 0;
//
//  // BXDF Public Data
//  const BXDFType type;
//};

//struct Fresnel {
//  uint32_t dielectric;
//  float3 etaI, etaT, k;
//};
//
//float3 FresnelEvaluate(float cosThetaI, const Fresnel fresnel) const {
//  if (fresnel.dielectric == 1) {
//    return FrDielectric(fresnel.cosThetaI, fresnel.etaI, fresnel.etaT);
//  }
//  return FrConductor(abs(fresnel.cosThetaI), fresnel.etaI, fresnel.etaT, fresnel.k);
//}

#ifdef COMPUTE_SHADER_SCOPE

float3 SpecularReflection_f(const BXDFStruct bxdf, const float3 wo, const float3 wi);
float SpecularReflection_Pdf(const BXDFStruct bxdf, const float3 wo, const float3 wi);
SampleStruct SpecularReflection_Sample_f(const BXDFStruct bxdf, const float3 wo, const float2 sample, BXDFType sampledType);

SampleStruct BXDF_Sample_f(const BXDFStruct bxdf, const float3 wo, const float2 sample, BXDFType sampledType);

float3 BXDF_rho(const BXDFStruct bxdf, const float3 wo, int nSamples, const float2 samples) {
  float3 r = 0;
  for (int i = 0; i < nSamples; ++i) {
      // Estimate one term of $\rho_\roman{hd}$
      float3 wi;
      Float pdf = 0;
      const SampleStruct sample = BXDF_Sample_f(w, u[i]);
      if (pdf > 0) r += f * AbsCosTheta(wi) / pdf;
  }
  return r / nSamples;
}

float3 SpecularReflection_f(const BXDFStruct bxdf, const float3 wo, const float3 wi) {
  return float3(0.f);
}

float SpecularReflection_Pdf(const BXDFStruct bxdf, const float3 wo, const float3 wi) {
  return 0.f;
}

SampleStruct SpecularReflection_Sample_f(const BXDFStruct bxdf, const float3 wo, const float2 sample, BXDFType sampledType) {
  SampleStruct ret;

  // Compute perfect specular reflection direction
  ret.wi = float(-wo.x, -wo.y, wo.z);
  ret.f = FresnelEvaluate(CosTheta(ret.wi)) * bxdf.R / AbsCosTheta(ret.wi);
  ret.pdf = 1.f;

  return ret;
}

float3 SpecularTransmission_f(const BXDFStruct bxdf, const float3 wo, const float3 wi) const {
  return float3(0.f);
}

float SpecularTransmission_Pdf(const BXDFStruct bxdf, const float3 wo, const float3 wi) const {
  return 0.f;
}

SampleStruct SpecularTransmission_Sample_f(const BXDFStruct bxdf, const float3 wo, const float2 sample, BXDFType sampledType) const {
  SampleStruct ret;

  // Figure out which $\eta$ is incident and which is transmitted
  const bool entering = CosTheta(wo) > 0.f;
  const float etaI = entering ? bxdf.etaA.x : bxdf.etaB.x;
  const float etaT = entering ? bxdf.etaB.x : bxdf.etaA.x;

  const RefractOutStruct refractOut = Refract(wo, Faceforward(float3(0.f, 0.f, 1.f), wo), etaI / etaT);
  // Compute ray direction for specular transmission
  if (!refractOut.valid) {
    return 0.f;
  }
  ret.wi = refractOut.wt;
  ret.pdf = 1.f;
  const float3 ft = T * (float3(1.) - FresnelEvaluate(CosTheta(ret.wi)), bxdf.fresnel);
  // Account for non-symmetry with transmission to different medium
  if (mode == TransportMode::Radiance) {
    ft *= (etaI * etaI) / (etaT * etaT);
  }
  ret.f = ft / AbsCosTheta(ret.wi);
  return ret;
}

float3 LambertianReflection_f(const BXDFStruct bxdf, const float3 wo, const float3 wi) const {
  return bxdf.reflection * 0.31830988618379067154f;
}

float LambertianReflection_Pdf(const BXDFStruct bxdf, const float3 wo, const float3 wi) const {
  return 0.f;
}

SampleStruct LambertianReflection_Sample_f(const BXDFStruct bxdf, const float3 wo, const float2 sample, BXDFType sampledType) const {
  SampleStruct ret;

  // Figure out which $\eta$ is incident and which is transmitted
  const bool entering = CosTheta(wo) > 0.f;
  const float etaI = entering ? bxdf.etaA.x : bxdf.etaB.x;
  const float etaT = entering ? bxdf.etaB.x : bxdf.etaA.x;

  const RefractOutStruct refractOut = Refract(wo, Faceforward(float3(0.f, 0.f, 1.f), wo), etaI / etaT);
  // Compute ray direction for specular transmission
  if (!refractOut.valid) {
    return 0.f;
  }
  ret.wi = refractOut.wt;
  ret.pdf = 1.f;
  const float3 ft = T * (float3(1.) - FresnelEvaluate(CosTheta(ret.wi)), bxdf.fresnel);
  // Account for non-symmetry with transmission to different medium
  if (mode == TransportMode::Radiance) {
    ft *= (etaI * etaI) / (etaT * etaT);
  }
  ret.f = ft / AbsCosTheta(ret.wi);
  return ret;
}

#else

class SpecularReflection : public BXDFS {
public:
  // SpecularReflection Public Methods
  SpecularReflection(const float3 R, const Fresnel fresnel)
  : type(BXDFType(BSDF_REFLECTION | BSDF_SPECULAR)), reflection(R), fresnel(fresnel) {}
};

class SpecularTransmission : public BXDFS {
public:
  // SpecularTransmission Public Methods
  SpecularTransmission(const float3 T, float etaA, float etaB, TransportMode mode)
  : type(BXDFType(BSDF_TRANSMISSION | BSDF_SPECULAR)),
  T(transmission),
  fresnel(createFresnelDielectric(etaA, etaB)),
  mode(mode) {}
};

class LambertianReflection : public BXDFS {
public:
  // LambertianReflection Public Methods
  LambertianReflection(const float3 R)
  : type(BXDFType(BSDF_REFLECTION | BSDF_DIFFUSE)), R(R) {}
};

//class FresnelSpecular : public BXDF {
//public:
//  // FresnelSpecular Public Methods
//  FresnelSpecular(const float3 R, const float3 T, float etaA,
//                  float etaB, TransportMode mode)
//  : BXDF(BXDFType(BSDF_REFLECTION | BSDF_TRANSMISSION | BSDF_SPECULAR)),
//  R(R),
//  T(T),
//  etaA(etaA),
//  etaB(etaB),
//  mode(mode) {}
//  float3 f(const float3 wo, const float3 wi) const {
//    return float3(0.f);
//  }
//  float3 Sample_f(const float3 wo, float3 *wi, const Point2f &u,
//                  float *pdf, BXDFType *sampledType) const;
//  float Pdf(const float3 wo, const float3 wi) const { return 0; }
//  std::string ToString() const;
//
//private:
//  // FresnelSpecular Private Data
//  const float3 R, T;
//  const float etaA, etaB;
//  const TransportMode mode;
//};

//class LambertianReflection : public BXDF {
//public:
//  // LambertianReflection Public Methods
//  LambertianReflection(const float3 R)
//  : BXDF(BXDFType(BSDF_REFLECTION | BSDF_DIFFUSE)), R(R) {}
//  float3 f(const float3 wo, const float3 wi) const;
//  float3 rho(const float3 , int, const Point2f *) const { return R; }
//  float3 rho(int, const Point2f *, const Point2f *) const { return R; }
//  std::string ToString() const;
//
//private:
//  // LambertianReflection Private Data
//  const float3 R;
//};

//class LambertianTransmission : public BXDF {
//public:
//  // LambertianTransmission Public Methods
//  LambertianTransmission(const float3 T)
//  : BXDF(BXDFType(BSDF_TRANSMISSION | BSDF_DIFFUSE)), T(T) {}
//  float3 f(const float3 wo, const float3 wi) const;
//  float3 rho(const float3 , int, const Point2f *) const { return T; }
//  float3 rho(int, const Point2f *, const Point2f *) const { return T; }
//  float3 Sample_f(const float3 wo, float3 *wi, const Point2f &u,
//                  float *pdf, BXDFType *sampledType) const;
//  float Pdf(const float3 wo, const float3 wi) const;
//  std::string ToString() const;
//
//private:
//  // LambertianTransmission Private Data
//  float3 T;
//};

//class OrenNayar : public BXDF {
//public:
//  // OrenNayar Public Methods
//  float3 f(const float3 wo, const float3 wi) const;
//  OrenNayar(const float3 R, float sigma)
//  : BXDF(BXDFType(BSDF_REFLECTION | BSDF_DIFFUSE)), R(R) {
//    sigma = Radians(sigma);
//    float sigma2 = sigma * sigma;
//    A = 1.f - (sigma2 / (2.f * (sigma2 + 0.33f)));
//    B = 0.45f * sigma2 / (sigma2 + 0.09f);
//  }
//  std::string ToString() const;
//
//private:
//  // OrenNayar Private Data
//  const float3 R;
//  float A, B;
//};

//class MicrofacetReflection : public BXDF {
//public:
//  // MicrofacetReflection Public Methods
//  MicrofacetReflection(const float3 R,
//                       MicrofacetDistribution *distribution, Fresnel *fresnel)
//  : BXDF(BXDFType(BSDF_REFLECTION | BSDF_GLOSSY)),
//  R(R),
//  distribution(distribution),
//  fresnel(fresnel) {}
//  float3 f(const float3 wo, const float3 wi) const;
//  float3 Sample_f(const float3 wo, float3 *wi, const Point2f &u,
//                  float *pdf, BXDFType *sampledType) const;
//  float Pdf(const float3 wo, const float3 wi) const;
//  std::string ToString() const;
//
//private:
//  // MicrofacetReflection Private Data
//  const float3 R;
//  const MicrofacetDistribution *distribution;
//  const Fresnel *fresnel;
//};
//
//class MicrofacetTransmission : public BXDF {
//public:
//  // MicrofacetTransmission Public Methods
//  MicrofacetTransmission(const float3 T,
//                         MicrofacetDistribution *distribution, float etaA,
//                         float etaB, TransportMode mode)
//  : BXDF(BXDFType(BSDF_TRANSMISSION | BSDF_GLOSSY)),
//  T(T),
//  distribution(distribution),
//  etaA(etaA),
//  etaB(etaB),
//  fresnel(etaA, etaB),
//  mode(mode) {}
//  float3 f(const float3 wo, const float3 wi) const;
//  float3 Sample_f(const float3 wo, float3 *wi, const Point2f &u,
//                  float *pdf, BXDFType *sampledType) const;
//  float Pdf(const float3 wo, const float3 wi) const;
//  std::string ToString() const;
//
//private:
//  // MicrofacetTransmission Private Data
//  const float3 T;
//  const MicrofacetDistribution *distribution;
//  const float etaA, etaB;
//  const FresnelDielectric fresnel;
//  const TransportMode mode;
//};
//
//class FresnelBlend : public BXDF {
//public:
//  // FresnelBlend Public Methods
//  FresnelBlend(const float3 Rd, const float3 Rs,
//               MicrofacetDistribution *distrib);
//  float3 f(const float3 wo, const float3 wi) const;
//  float3 SchlickFresnel(float cosTheta) const {
//    auto pow5 = [](float v) { return (v * v) * (v * v) * v; };
//    return Rs + pow5(1 - cosTheta) * (float3(1.) - Rs);
//  }
//  float3 Sample_f(const float3 wi, float3 *sampled_f, const Point2f &u,
//                  float *pdf, BXDFType *sampledType) const;
//  float Pdf(const float3 wo, const float3 wi) const;
//  std::string ToString() const;
//
//private:
//  // FresnelBlend Private Data
//  const float3 Rd, Rs;
//  MicrofacetDistribution *distribution;
//};

//class FourierBSDF : public BXDF {
//public:
//  // FourierBSDF Public Methods
//  float3 f(const float3 wo, const float3 wi) const;
//  FourierBSDF(const FourierBSDFTable &bsdfTable, TransportMode mode)
//  : BXDF(BXDFType(BSDF_REFLECTION | BSDF_TRANSMISSION | BSDF_GLOSSY)),
//  bsdfTable(bsdfTable),
//  mode(mode) {}
//  float3 Sample_f(const float3 wo, float3 *wi, const Point2f &u,
//                  float *pdf, BXDFType *sampledType) const;
//  float Pdf(const float3 wo, const float3 wi) const;
//  std::string ToString() const;
//
//private:
//  // FourierBSDF Private Data
//  const FourierBSDFTable &bsdfTable;
//  const TransportMode mode;
//};

//// BSDF Inline Method Definitions
//int NumComponents(BXDFType flags) const {
//  int num = 0;
//  for (int i = 0; i < nBXDFs; ++i)
//    if (bxdfs[i]->MatchesFlags(flags)) ++num;
//  return num;
//}

//#endif

//class BSDF : public BSDFStruct{
//public:
//  // BSDF Public Methods
//  BSDF(const SurfaceInteraction &si, float eta = 1.f)
//  : eta(eta),
//  ns(si.shading.n),
//  ng(si.n),
//  ss(Normalize(si.shading.dpdu)),
//  ts(Cross(ns, ss)) {}
//  void Add(BXDF *b) {
//    CHECK_LT(nBXDFs, MaxBXDFs);
//    bxdfs[nBXDFs++] = b;
//  }
//  int NumComponents(BXDFType flags = BSDF_ALL) const;
//
//  float3 f(const float3 woW, const float3 wiW,
//           BXDFType flags = BSDF_ALL) const;
//  float3 rho(int nSamples, const Point2f *samples1, const Point2f *samples2,
//             BXDFType flags = BSDF_ALL) const;
//  float3 rho(const float3 wo, int nSamples, const Point2f *samples,
//             BXDFType flags = BSDF_ALL) const;
//  float3 Sample_f(const float3 wo, float3 *wi, const Point2f &u,
//                  float *pdf, BXDFType type = BSDF_ALL,
//                  BXDFType *sampledType = nullptr) const;
//  float Pdf(const float3 wo, const float3 wi,
//            BXDFType flags = BSDF_ALL) const;
//  std::string ToString() const;
//
//  // BSDF Public Data
//  const float eta;
//
//private:
//  // BSDF Private Methods
//  ~BSDF() {}
//
//  // BSDF Private Data
////  const float3 ns, ng;
////  const float3 ss, ts;
////  int nBXDFs = 0;
//  static PBRT_CONSTEXPR int MaxBXDFs = 8;
//  BXDF *bxdfs[MaxBXDFs];
//  friend class MixMaterial;
//};

//SampleStruct SpecularTransmission_Sample_f(const BXDFStruct bxdf, const float3 wo, const float2 sample, BXDFType sampledType) const {
//  SampleStruct ret;
//
//  // Cosine-sample the hemisphere, flipping the direction if necessary
//  ret.wi = CosineSampleHemisphere(u);
//  if (wo.z < 0.f) {
//    ret.wi.z *= -1.f;
//  }
//  ret.pdf = SpecularTransmission_Pdf(wo, ret.wi);
//  ret.f = SpecularTransmission_f(wo, ret.wi);
//  return ret;
//}

#endif
#endif
