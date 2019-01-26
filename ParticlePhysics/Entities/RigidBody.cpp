#include "RigidBody.h"

RigidBody::RigidBody()
{
  solver = SOLVER_RIGID_BODY;
}

void RigidBody::initCube(const real dimensions[], const real particleRadius, const real mass)
{
  vector<uint32_t> connectionElements;
  uint subdivision[3];

  for (uint i = 0; i < 3; i++)
  {
    subdivision[i] = (uint)(dimensions[i] / (2.f * particleRadius));
  }

  Real3 del[3];
  Real3 offset;
  for (uint i = 0; i < 3; i++)
  {
    del[i] = 0.f;
    del[i][i] = 2 * particleRadius;
  }
  del[1] *= -1;
  del[2] *= -1;

  Real3 top_left(particleRadius - dimensions[0] / 2.f, dimensions[1] / 2.f - particleRadius, dimensions[2] / 2.f - particleRadius);

  vector<Real3> pointPosition;
  int signedSubdivision[] = { subdivision[0], subdivision[1], subdivision[2] };

  const real perParticleInvMass = real(mass) / real(subdivision[0] * subdivision[1] * subdivision[2]);

  (*entitySharedData.host())[0].invMassIsShared = 1;
  (*entitySharedData.host())[0].sharedInvMass = perParticleInvMass;
  (*entitySharedData.host())[0].radiusIsShared = 1;
  (*entitySharedData.host())[0].sharedRadius = particleRadius;
  (*entitySharedData.host())[0].collisionDataIsShared = 0;

  for (int z = 0; z < signedSubdivision[2]; z++)
  {
    for (int y = 0; y < signedSubdivision[1]; y++)
    {
      int index = z*signedSubdivision[1] * signedSubdivision[0] + y*signedSubdivision[0];
      for (int x = 0; x < signedSubdivision[0]; x++)
      {
        Real3 newPosition = top_left + del[0] * (float)x + del[1] * (float)y + del[2] * (float)z;
        setConstant(index, newPosition);
        pointPosition.push_back(newPosition);

        float nx, ny, nz;
        nx = (x == 0) ? -1.f : (x == (signedSubdivision[0] - 1) ? 1.f : 0);
        ny = (y == 0) ? 1.f : (y == (signedSubdivision[1] - 1) ? -1.f : 0);
        nz = (z == 0) ? 1.f : (z == (signedSubdivision[2] - 1) ? -1.f : 0);
        Real3 normal(nx, ny, nz);

#ifdef ENABLE_RENDERING
        for (int zn = -1; zn < 2; zn++)
        {
          int tempz = z + zn;
          if (tempz < 0 || tempz >= signedSubdivision[2]) continue;
          for (int yn = -1; yn < 2; yn++)
          {
            int tempy = y + yn;
            if (tempy < 0 || tempy >= signedSubdivision[1]) continue;
            for (int xn = -1; xn < 2; xn++)
            {
              int tempx = x + xn;
              if (tempx < 0 || tempx >= signedSubdivision[0]) continue;

              int index1 = tempx + (tempy*signedSubdivision[0]) + (tempz*signedSubdivision[0] * signedSubdivision[1]);

              if (index == index1) continue;

              connectionElements.push_back(index);
              connectionElements.push_back(index1);
            }
          }
        }
#endif

        ParticleCollisionData colData;
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
        float magnitude = normal.length()*particleRadius;
        if (magnitude > 0)
        {
          normal.normalize();
        }
        colData.initialSdfGradient = normal;// *particleRadius;// Real3(nx, ny, nz);
        colData.sdfMagnitude = magnitude;

        //auxData.sdfGradient *= -1;
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

#ifdef ENABLE_RENDERING
  displayVertex.gen();
  displayVertex.copyData(&pointPosition[0][0], subdivision[0] * subdivision[1] * subdivision[2], 0, sizeof(Real3));

  displayElements.gen();
  displayElements.copyData((GLuint*)&connectionElements[0], connectionElements.size());

  displayShader.init("SolidVert.glsl", "SolidFrag.glsl");
#endif
}

#ifdef ENABLE_RENDERING

void RigidBody::render(ParticleStruct* particles)
{
  GLfloat model_mat[16], proj_mat[16];
  glGetFloatv(GL_PROJECTION_MATRIX, proj_mat);
  glGetFloatv(GL_MODELVIEW_MATRIX, model_mat);

  glEnable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);

  glPushMatrix();
  displayShader.bind();
  displayShader.set("modelViewMatrix", model_mat);
  displayShader.set("projectionMatrix", proj_mat);

  GL_CHECK(glEnableVertexAttribArray(0));
  GL_CHECK(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(ParticleStruct), particles));
  displayElements.bind();
  GL_CHECK(glDrawElementsInstanced(GL_TRIANGLES, displayElements.count(), GL_UNSIGNED_INT, 0, 1));
  displayElements.unbind();
  GL_CHECK(glDisableVertexAttribArray(0));

  displayShader.unbind();
  glPopMatrix();
}

#endif