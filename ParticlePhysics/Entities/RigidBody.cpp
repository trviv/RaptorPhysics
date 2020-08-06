#include "RigidBody.h"

RigidBody::RigidBody()
{
  solver = SOLVER_RIGID_BODY;
}

void RigidBody::initCube(const real dimensions[], real particleRadius, const real mass)
{
  vector<uint32_t> connectionElements;
  vector<uint32_t> endIndices;

  uint subdivision[3];
  const int density = 2;
  const bool reducedParticles = true;

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

        float magnitude = normal.length()*particleRadius;

        if (reducedParticles && magnitude == 0.f)
        {
          continue;
        }

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

        if (reducedParticles && magnitude == 0.f)
        {
          continue;
        }

        Real3 newPosition = top_left + del[0] * (float)x + del[1] * (float)y + del[2] * (float)z;
        setConstant(index, newPosition);
        pointPosition.push_back(newPosition);

        if ((x == 0 || x == (signedSubdivision[0] - 1)) &&
            (y == 0 || y == (signedSubdivision[1] - 1)) &&
            (z == 0 || z == (signedSubdivision[2] - 1)))
        {
          endIndices.push_back(index);
        }

        /*if ((normal[0] == normal[1]) && (normal[0] == normal[2]) && (normal[0] == 0))
        {
        Real3 axis(x - signedSubdivision[0] / 2, y - signedSubdivision[1] / 2, z - signedSubdivision[2] / 2);
        //axis *= -1;
        normal = axis;
        //normal[axis.longestAxis()] = 0;
        if (normal[0] == normal[1] == normal[2] == 0)
        {
        normal[0] = 1;
        }
        }*/
        if (magnitude > 0)
        {
          normal.normalize();
        }

        ParticleCollisionData colData;
        colData.transformedSdfGradient = 0;
        colData.gradientMagnitude = 0.f;
        colData.invMass = perParticleInvMass;
        colData.radius = particleRadius;
        particleCollisionData.host()->push_back(colData);

        ParticleRigidData rigidData;
        rigidData.initialSdfGradient = normal;
        rigidData.gradientMagnitude = particleRadius;
        particleRigidData.host()->push_back(rigidData);

        index++;
      }
    }
  }

  for (int i=0; i<2; i++)
  {
    connectionElements.push_back(endIndices[i * 4 + 0]);
    connectionElements.push_back(endIndices[i * 4 + 1]);
    connectionElements.push_back(endIndices[i * 4 + 2]);
    connectionElements.push_back(endIndices[i * 4 + 2]);
    connectionElements.push_back(endIndices[i * 4 + 1]);
    connectionElements.push_back(endIndices[i * 4 + 3]);

    connectionElements.push_back(endIndices[i * 2 + 0]);
    connectionElements.push_back(endIndices[i * 2 + 1]);
    connectionElements.push_back(endIndices[i * 2 + 4]);
    connectionElements.push_back(endIndices[i * 2 + 4]);
    connectionElements.push_back(endIndices[i * 2 + 1]);
    connectionElements.push_back(endIndices[i * 2 + 5]);

    connectionElements.push_back(endIndices[i + 0]);
    connectionElements.push_back(endIndices[i + 2]);
    connectionElements.push_back(endIndices[i + 4]);
    connectionElements.push_back(endIndices[i + 4]);
    connectionElements.push_back(endIndices[i + 2]);
    connectionElements.push_back(endIndices[i + 6]);
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
    particleRigidData.host()->at(i).initialComOffset = (*points)[i] - com;
  }

#ifdef ENABLE_RENDERING
  displayElements.gen();
  displayElements.copyData((GLuint*)&connectionElements[0], (uint)connectionElements.size());

  displayEdges.gen();
  uint boxIndices[] = {
    endIndices[0], endIndices[1], endIndices[0], endIndices[2], endIndices[0], endIndices[4],
    endIndices[1], endIndices[3], endIndices[1], endIndices[5],
    endIndices[2], endIndices[3], endIndices[2], endIndices[6],
    endIndices[3], endIndices[7],
    endIndices[4], endIndices[5], endIndices[4], endIndices[6],
    endIndices[5], endIndices[7],
    endIndices[6], endIndices[7]
  };
  displayEdges.copyData(boxIndices, 24);
#endif
}
