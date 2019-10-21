#include "Fluid.h"

Fluid::Fluid()
{
  solver = SOLVER_FLUID;
}

void Fluid::initFluid(const real dimensions[], real particleRadius, const real mass, float kernelRadius)
{
  uint subdivision[3];
  const int density = 1;

  for (uint i = 0; i < 3; i++)
  {
    subdivision[i] = (uint)(dimensions[i] / (2.f * particleRadius));
  }

  Real3 del[3];
  Real3 offset;
  for (uint i = 0; i < 3; i++)
  {
    del[i] = 0.f;
    del[i][i] = 2.f * particleRadius / density;
  }
  del[1] *= -1;
  del[2] *= -1;

  Real3 top_left(particleRadius - dimensions[0] / 2.f, dimensions[1] / 2.f - particleRadius, dimensions[2] / 2.f - particleRadius);

  vector<Real3> pointPosition;
  int signedSubdivision[] = { density * (int)subdivision[0], density * (int)subdivision[1], density * (int)subdivision[2] };

  int particleCount = 0;

  // calculate particle count before hand for proper weight calculation
  for (int z = 0; z < signedSubdivision[2]; z++)
  {
    for (int y = 0; y < signedSubdivision[1]; y++)
    {
      for (int x = 0; x < signedSubdivision[0]; x++)
      {
        float nx, ny, nz;
        nx = (x == 0) ? -1.f : (x == (signedSubdivision[0] - 1) ? 1.f : 0);
        ny = (y == 0) ? 1.f : (y == (signedSubdivision[1] - 1) ? -1.f : 0);
        nz = (z == 0) ? 1.f : (z == (signedSubdivision[2] - 1) ? -1.f : 0);
        Real3 normal(nx, ny, nz);

        particleCount++;
      }
    }
  }

  int index = 0;

  const real perParticleInvMass = real(particleCount) / real(mass);

  setInvMassIsShared((*entitySharedData.host())[0], true);
  (*entitySharedData.host())[0].sharedInvMass = perParticleInvMass;
  setRadiusIsShared((*entitySharedData.host())[0], true);
  (*entitySharedData.host())[0].sharedRadius = particleRadius;
  setCollisionDataIsShared((*entitySharedData.host())[0], false);
  (*entitySharedData.host())[0].fluidKernelRadius = kernelRadius;

  for (int z = 0; z < signedSubdivision[2]; z++)
  {
    for (int y = 0; y < signedSubdivision[1]; y++)
    {
      for (int x = 0; x < signedSubdivision[0]; x++)
      {
        float nx, ny, nz;
        nx = (x == 0) ? -1.f : (x == (signedSubdivision[0] - 1) ? 1.f : 0);
        ny = (y == 0) ? 1.f : (y == (signedSubdivision[1] - 1) ? -1.f : 0);
        nz = (z == 0) ? 1.f : (z == (signedSubdivision[2] - 1) ? -1.f : 0);
        Real3 normal(nx, ny, nz);

        float magnitude = normal.length() * particleRadius;

        Real3 newPosition = top_left + del[0] * (float)x + del[1] * (float)y + del[2] * (float)z;
        setConstant(index, newPosition);
        pointPosition.push_back(newPosition);

        ParticleCollisionData colData;

        if (magnitude > 0)
        {
          normal.normalize();
        }
        colData.initialSdfGradient = normal * particleRadius;
        colData.radius = particleRadius;
        colData.invMass = perParticleInvMass;

        particleCollisionData.host()->push_back(colData);

        ParticleAuxData auxData;
        auxData.invMass = perParticleInvMass;
        auxData.radius = particleRadius;
        particleAuxData.host()->push_back(auxData);

        index++;
      }
    }
  }

  Real3 com(0);
  uint prev_value_count = 0;
  vector<Real3>* points = constrainConstants.host();

  for (uint i = prev_value_count; i < points->size(); i++)
  {
    com += (*points)[i];
  }

  com /= real(points->size() - prev_value_count);

  for (uint i = prev_value_count; i < points->size(); i++)
  {
    ParticleRigidData rigidData;
    rigidData.initialComOffset = (*points)[i] - com;
    particleRigidData.host()->push_back(rigidData);
  }
}
