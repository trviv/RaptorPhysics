#include "Cloth.h"
#include "PhysicsSystem.h"

Cloth::Cloth()
{
  solver = SOLVER_CLOTH;
}

void Cloth::initXY(const real dimensions[], const real particleRadius, const real mass)
{
  uint subdivision[3] = { 1, 1, 1 };
  for (int i = 0; i < 2; i++) subdivision[i] = uint(dimensions[i] / particleRadius);
  (*entitySharedData.host())[0].sharedRadius = particleRadius;
  initXY(dimensions, subdivision, mass);
}

void Cloth::initXY(const real dimensions[], const uint subdivision[], const real mass)
{
  vector<uint32_t> connectionElements;
  vector<uint32_t> edgeElements;

  Real3 del_x(0);
  if (subdivision[0] > 1)
  {
    del_x[0] = dimensions[0] / (subdivision[0] - 1);
  }
  Real3 del_y(0);
  if (subdivision[1] > 1)
  {
    del_y[1] = -dimensions[1] / (subdivision[1] - 1);
  }
  Real3 top_left(-dimensions[0] / 2, dimensions[1] / 2, 0);

  const real x_len = del_x.length();
  const real y_len = del_y.length();
  const real diag_len = mSqrt(mSqr(x_len) + mSqr(y_len));
  vector<Real3> pointPosition;

  const real perParticleInvMass = real(subdivision[0] * subdivision[1]) / real(mass);
  const float minRadius = (x_len < y_len ? x_len : y_len) / 2;

  setRadiusIsShared((*entitySharedData.host())[0], true);
  (*entitySharedData.host())[0].sharedRadius = minRadius;
  setInvMassIsShared((*entitySharedData.host())[0], false);

  for (uint y = 0; y < subdivision[1]; y++)
  {
    uint index = y * subdivision[0];
    for (uint x = 0; x < subdivision[0]; x++)
    {
      addConnection(index, index, y || (x > 0 && x < subdivision[0]-1) ? perParticleInvMass : 0);
      ParticleAuxData auxData;
      auxData.invMass = rawConstrainCoefficients[index][0];
      auxData.radius = minRadius;
      particleAuxData.host()->push_back(auxData);
      index++;
    }
  }

  for (uint y = 0; y < subdivision[1]; y++)
  {
    Real3 pos = top_left + del_y * (float)y;
    uint index = y * subdivision[0];
    for (uint x = 0; x < subdivision[0]; x++)
    {
      Real3 newPosition = pos + Real3(0, 0, (((subdivision[1] - y) == 1 && (subdivision[0] - x) == 1) ? .5f : 0));
      setConstant(index, newPosition);
      pointPosition.push_back(newPosition);
      ParticleCollisionData colData;
      colData.initialSdfGradient = Real3(0.f);
      colData.radius = minRadius;
      colData.transformedSdfGradient = Real3(0.f);
      colData.invMass = particleAuxData.host()->at(index).invMass;
      particleCollisionData.host()->push_back(colData);

      // add twice because constrain is solved only once
      if (x + 1 < subdivision[0])
      {
        addConnection(index, index + 1, x_len);
        addConnection(index + 1, index, x_len);
      }
      if (y + 1 < subdivision[1])
      {
        addConnection(index, index + subdivision[0], y_len);
        addConnection(index + subdivision[0], index, y_len);

        if (false && x + 1 >= subdivision[0])
        {
          connectionElements.push_back(index);
          connectionElements.push_back(index + subdivision[0]);
          connectionElements.push_back(index + subdivision[0] - 1);
        }
      }
      if (x + 1 < subdivision[0] && y + 1 < subdivision[1])
      {
        addConnection(index, index + subdivision[0] + 1, diag_len);
        addConnection(index + subdivision[0] + 1, index, diag_len);

        connectionElements.push_back(index);
        connectionElements.push_back(index + subdivision[0]);
        connectionElements.push_back(index + subdivision[0] + 1);

        connectionElements.push_back(index);
        connectionElements.push_back(index + subdivision[0] + 1);
        connectionElements.push_back(index + 1);

        edgeElements.push_back(index);
        edgeElements.push_back(index + 1);

        edgeElements.push_back(index);
        edgeElements.push_back(index + subdivision[0]);
      }
      if (x >= 1 && y + 1 < subdivision[1])
      {
        addConnection(index, index + subdivision[0] - 1, diag_len);
        addConnection(index + subdivision[0] - 1, index, diag_len);
      }

      index++;
      pos += del_x;
    }
  }

#ifdef ENABLE_RENDERING
  displayElements.gen();
  displayElements.copyData((GLuint*)&connectionElements[0], (uint)connectionElements.size());

  displayEdges.gen();
  displayEdges.copyData((GLuint*)&edgeElements[0], (uint)edgeElements.size());
#endif
}
