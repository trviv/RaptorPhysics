/*
 * Copyright (c) Vivek Trivedi, 2025
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#ifndef BXDF_COMMON_H
#define BXDF_COMMON_H

enum class TransportMode { Radiance, Importance };

// BSDF Declarations
enum BXDFType {
  BSDF_REFLECTION = 1 << 0,
  BSDF_TRANSMISSION = 1 << 1,
  BSDF_DIFFUSE = 1 << 2,
  BSDF_GLOSSY = 1 << 3,
  BSDF_SPECULAR = 1 << 4,
  BSDF_ALL = BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_SPECULAR | BSDF_REFLECTION |
  BSDF_TRANSMISSION,
};

struct Fresnel {
  uint32_t dielectric;
  float3 etaI, etaT, k;
};

#ifndef COMPUTE_SHADER_SCOPE

static Fresnel createFresnelConductor(const float3 etaI, const float3 etaT, const float3 k) {
  return {0, etaI, etaT, k};
}

static Fresnel createFresnelDielectric(const float etaA, const float etaB) {
  return {1, Real3(etaA), Real3(etaB), Real3(0.f)};
}

#else

#include "ComputeHeader.shader"
#include "ComputeShared.h"

// Reflection Declarations
float FrDielectric(float cosThetaI, float etaI, float etaT) {
  cosThetaI = clamp(cosThetaI, -1.f, 1.f);
  // Potentially swap indices of refraction
  bool entering = cosThetaI > 0.f;
  if (!entering) {
    swapFloat(etaI, etaT);
    cosThetaI = abs(cosThetaI);
  }

  // Compute _cosThetaT_ using Snell's law
  float sinThetaI = sqrt(max(0.f, 1.f - cosThetaI * cosThetaI));
  float sinThetaT = etaI / etaT * sinThetaI;

  // Handle total internal reflection
  if (sinThetaT >= 1.f) {
    return 1.f;
  }

  float cosThetaT = sqrt(max(0.f, 1.f - sinThetaT * sinThetaT));
  float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
  float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));
  return (Rparl * Rparl + Rperp * Rperp) / 2.f;
}

// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
float3 FrConductor(float cosThetaI, const float3 etai,
                   const float3 etat, const float3 k) {
  cosThetaI = clamp(cosThetaI, -1.f, 1.f);
  const float3 eta = etat / etai;
  const float3 etak = k / etai;

  float cosThetaI2 = cosThetaI * cosThetaI;
  float sinThetaI2 = 1. - cosThetaI2;
  const float3 eta2 = eta * eta;
  const float3 etak2 = etak * etak;

  const float3 t0 = eta2 - etak2 - sinThetaI2;
  const float3 a2plusb2 = sqrt(t0 * t0 + 4 * eta2 * etak2);
  const float3 t1 = a2plusb2 + cosThetaI2;
  const float3 a = sqrt(0.5f * (a2plusb2 + t0));
  const float3 t2 = 2.f * cosThetaI * a;
  const float3 Rs = (t1 - t2) / (t1 + t2);

  float3 t3 = cosThetaI2 * a2plusb2 + sinThetaI2 * sinThetaI2;
  float3 t4 = t2 * sinThetaI2;
  float3 Rp = Rs * (t3 - t4) / (t3 + t4);

  return 0.5f * (Rp + Rs);
}

// BSDF Inline Functions
float CosTheta(const float3 w) { return w.z; }
float Cos2Theta(const float3 w) { return w.z * w.z; }
float AbsCosTheta(const float3 w) { return abs(w.z); }
float Sin2Theta(const float3 w) {
  return max(0.f, 1.f - Cos2Theta(w));
}

float SinTheta(const float3 w) { return sqrt(Sin2Theta(w)); }

float TanTheta(const float3 w) { return SinTheta(w) / CosTheta(w); }

float Tan2Theta(const float3 w) {
  return Sin2Theta(w) / Cos2Theta(w);
}

float CosPhi(const float3 w) {
  const float sinTheta = SinTheta(w);
  return (sinTheta == 0.f) ? 1.f : clamp(w.x / sinTheta, -1.f, 1.f);
}

float SinPhi(const float3 w) {
  const float sinTheta = SinTheta(w);
  return (sinTheta == 0.f) ? 0.f : clamp(w.y / sinTheta, -1.f, 1.f);
}

float Cos2Phi(const float3 w) { return CosPhi(w) * CosPhi(w); }

float Sin2Phi(const float3 w) { return SinPhi(w) * SinPhi(w); }

float CosDPhi(const float3 wa, const float3 wb) {
  const float waxy = wa.x * wa.x + wa.y * wa.y;
  const float wbxy = wb.x * wb.x + wb.y * wb.y;
  if (waxy == 0.f || wbxy == 0.f) {
    return 1.f;
  }
  return clamp((wa.x * wb.x + wa.y * wb.y) / sqrt(waxy * wbxy), -1.f, 1.f);
}

float3 Reflect(const float3 wo, const float3 n) {
  return -wo + 2.f * dot(wo, n) * n;
}

struct RefractOutStruct {
  bool valid;
  float3 wt;
};

RefractOutStruct Refract(const float3 wi, const float3 n, float eta) {
  // Compute $\cos \theta_\roman{t}$ using Snell's law
  const float cosThetaI = dot(n, wi);
  const float sin2ThetaI = max(0.f, 1.f - cosThetaI * cosThetaI);
  const float sin2ThetaT = eta * eta * sin2ThetaI;

  RefractOutStruct ret;
  // Handle total internal reflection for transmission
  if (sin2ThetaT >= 1.f) {
    ret.valid = false;
    return ret;
  }

  const float cosThetaT = sqrt(1 - sin2ThetaT);
  ret.wt = eta * -wi + (eta * cosThetaI - cosThetaT) * float3(n);
  ret.valid = true;
  return ret;
}

bool SameHemisphere(const float3 w, const float3 wp) {
  return w.z * wp.z > 0;
}

float3 FresnelEvaluate(float cosThetaI, const Fresnel fresnel) {
  if (fresnel.dielectric == 1) {
    return FrDielectric(cosThetaI, fresnel.etaI.x, fresnel.etaT.x);
  }
  return FrConductor(abs(cosThetaI), fresnel.etaI, fresnel.etaT, fresnel.k);
}

float3 Faceforward(const float3 n, const float3 v) {
    return (dot(n, v) < 0.f) ? -n : n;
}

#endif
#endif
