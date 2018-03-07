#include "RigidBody.h"

RigidBody::RigidBody()
{
  solver = SOLVER_RIGID_BODY;
}

void RigidBody::initCube(const real dimensions[], const real particleRadius, const real mass)
{
  uint subdivision[3];
  for (uint i = 0; i < 3; i++)
  {
    subdivision[i] = (uint)(dimensions[i] / particleRadius);
  }
  initCube(dimensions, subdivision, mass);
}

void RigidBody::initCube(const real dimensions[], const uint subdivision[], const real mass)
{
  vector<uint32_t> connectionElements;

  Real3 del[3];
  for (uint i = 0; i < 3; i++)
  {
    del[i] = 0;
    del[i][i] = real(2.)*dimensions[i] / (subdivision[i] - 1);
  }
  del[1] *= -1;
  del[2] *= -1;
  Real3 top_left(-dimensions[0], dimensions[1], dimensions[2]);

  vector<Real3> pointPosition;
  int signedSubdivision[] = { subdivision[0], subdivision[1], subdivision[2] };

  const real perParticleInvMass = real(mass) / real(subdivision[0] * subdivision[1] * subdivision[2]);

  (*particleSharedData.host())[0].invMassIsShared = 1;
  (*particleSharedData.host())[0].sharedInvMass = perParticleInvMass;

  uint prev_value_count = 0;

  for (int z = 0; z < signedSubdivision[2]; z++)
  {
    for (int y = 0; y < signedSubdivision[1]; y++)
    {
      int index = z*signedSubdivision[1] * signedSubdivision[0] + y*signedSubdivision[0];
      for (int x = 0; x < signedSubdivision[0]; x++)
      {
        Real3 newPosition = top_left + del[0] * x + del[1] * y + del[2] * z;
        setConstant(index, newPosition);
        pointPosition.push_back(newPosition);

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
              Real3 pos2 = top_left + (del[0] * tempx) + (del[1] * tempy) + (del[2] * tempz);

              if (index1 == index) continue;

              if (index1 > index)
              {
                connectionElements.push_back(index);
                connectionElements.push_back(index1);
              }
            }
          }
        }
        index++;
      }
    }
  }

  Real3 com(0);
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
  (*points)[prev_value_count] -= real(1.5);

#ifdef ENABLE_RENDERING
  displayVertex.gen();
  displayVertex.copyData(&pointPosition[0][0], subdivision[0] * subdivision[1] * subdivision[2], 0, sizeof(Real3));

  displayElements.gen();
  displayElements.copyData((GLuint*)&connectionElements[0], connectionElements.size());

  displayShader.init("display_vert.glsl", "display_frag.glsl");
#endif
}

#ifdef ENABLE_RENDERING

void RigidBody::render(ParticleStruct* particles)
{
  GLfloat model_mat[16], proj_mat[16];
  glGetFloatv(GL_PROJECTION_MATRIX, proj_mat);
  glGetFloatv(GL_MODELVIEW_MATRIX, model_mat);

  glDisable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);
  glDisable(GL_LIGHTING);
  glDisable(GL_POINT_SMOOTH);
  glDisable(GL_CULL_FACE);

  glPushMatrix();
  displayShader.bind();
  displayShader.set("modelViewMatrix", model_mat);
  displayShader.set("projectionMatrix", proj_mat);
  //displayVertex.bind();
  GL_CHECK(glEnableVertexAttribArray(0));
  GL_CHECK(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(ParticleStruct), particles));
  //GL_CHECK(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Real3), (GLvoid*)0));
  displayElements.bind();
  GL_CHECK(glDrawElements(GL_LINES, displayElements.count(), GL_UNSIGNED_INT, NULL));
  displayElements.unbind();
  GL_CHECK(glDisableVertexAttribArray(0));
  //displayVertex.unbind();
  displayShader.unbind();
  glPopMatrix();
}

#endif