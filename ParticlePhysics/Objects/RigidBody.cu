#include "RigidBody.h"

void RigidBody::init(const Matrix4& transform, const real dim[],
  const real particle_radius, const real mass)
{
  Counter subdivision[3];
  for (Counter i = 0; i < 3; i++) subdivision[i] = dim[i] / particle_radius;
  this->particle_radius = particle_radius;
  init(transform, dim, subdivision, mass);
}

void RigidBody::init(const Matrix4& transform, const real dim[],
  const Counter subdivision[], const real mass)
{
  Real3 del[3];
  for (Counter i = 0; i < 3; i++)
  {
    del[i] = 0;
    del[i][i] = real(2.)*dim[i] / (subdivision[i] - 1);
  }
  del[1] *= -1;
  del[2] *= -1;
  Real3 top_left(-dim[0], dim[1], dim[2]);

  std::vector<Real3> point_pos;
  std::vector<Real3> &constrain_values = physics_system->rigidConstrain()->hostPosition();
  std::vector<Real3> &com_offset = physics_system->rigidConstrain()->hostComOffset();
  Counter prev_value_count = constrain_values.size();

  for (Counter z = 0; z < subdivision[2]; z++)
  {
    for (Counter y = 0; y < subdivision[1]; y++)
    {
      Counter index = (z*subdivision[2] * subdivision[1]) + (y*subdivision[1]);
      for (Counter x = 0; x < subdivision[0]; x++)
      {
        //physics_system->rigidConstrain()->addDistance(index, index, 0);
        index++;
      }
    }
  }

  for (Counter z = 0; z < subdivision[2]; z++)
  {
    for (Counter y = 0; y < subdivision[1]; y++)
    {
      Counter index = z*subdivision[1] * subdivision[0] + y*subdivision[0];
      for (Counter x = 0; x < subdivision[0]; x++)
      {
        Real3 pos = top_left + del[0] * x + del[1] * y + del[2] * z;
        physics_system->rigidConstrain()->setValue(index, pos);
        point_pos.push_back(pos);
        for (Counter zn = -1; zn < 2; zn++)
        {
          Counter tempz = z + zn;
          if (tempz < 0 || tempz >= subdivision[2]) continue;
          for (Counter yn = -1; yn < 2; yn++)
          {
            Counter tempy = y + yn;
            if (tempy < 0 || tempy >= subdivision[1]) continue;
            for (Counter xn = -1; xn < 2; xn++)
            {
              Counter tempx = x + xn;
              if (tempx < 0 || tempx >= subdivision[0]) continue;
              Counter index1 = tempx + (tempy*subdivision[0]) +
                (tempz*subdivision[0] * subdivision[1]);
              Real3 pos2 = top_left + (del[0] * tempx) + (del[1] * tempy)
                + (del[2] * tempz);
              if (index1 == index) continue;
              real len = (pos - pos2).length();
              physics_system->rigidConstrain()->addDistance(index, index1, len);
              physics_system->rigidConstrain()->setInvMass(1);
              if (index1 > index)
              {
                connection_elements.push_back(index);
                connection_elements.push_back(index1);
              }
              //physics_system->rigidConstrain()->add(index1, index, 1, len);
            }
          }
        }
        index++;
      }
    }
  }

  Real3 com(0);
  for (Counter i = prev_value_count; i < constrain_values.size(); i++)
    com += constrain_values[i];
  com /= real(constrain_values.size() - prev_value_count);
  for (Counter i = prev_value_count; i < constrain_values.size(); i++)
    com_offset.push_back(constrain_values[i] - com);

  constrain_values[prev_value_count] -= real(.5);

  disp_vertex.gen();
  disp_vertex.copyData(&point_pos[0][0],
    subdivision[0] * subdivision[1] * subdivision[2], 0, sizeof(Real3));
  disp_elements.gen();
  disp_elements.copyData((GLuint*)&connection_elements[0],
    connection_elements.size());
  disp_shader.init("../../ParticlePhysics/display_vert.glsl",
    "../../ParticlePhysics/display_frag.glsl");
  plug.setGLResource(disp_vertex);
}

void RigidBody::step()
{
  physics_system->rigidConstrain()->solve();
  CU_PROMPT;
  void* vertex_array = this->plug.map();

  dim3 threads;
  dim3 blocks;
  physics_system->rigidConstrain()->configureGrid(threads, blocks);

  DeviceEntity<Real3>::copy((Real3*)vertex_array,
    (Real3*)physics_system->rigidConstrain()->getPosition(),
    physics_system->rigidConstrain()->getNodeCount());
  CU_PROMPT;
  this->plug.unmap();
}

void RigidBody::render()
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
  disp_shader.bind();
  disp_shader.set("modelViewMatrix", model_mat);
  disp_shader.set("projectionMatrix", proj_mat);
  disp_vertex.bind();
  GL_CHECK(glEnableVertexAttribArray(0));
  GL_CHECK(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Real3), (GLvoid*)0));
  //GL_CHECK(glEnableVertexAttribArray(1));
  //GL_CHECK(glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (GLvoid*)0));
  //GL_CHECK(glDrawArrays(GL_POINTS, 0, disp_vertex.count()));
  //GL_CHECK(glDrawElementsInstanced(GL_TRIANGLES, grid_mesh_face.count(), GL_UNSIGNED_INT, 0, pow(simulation_dim[0] - 1, 2)));
  disp_elements.bind();
  GL_CHECK(glDrawElements(GL_LINES, disp_elements.count(), GL_UNSIGNED_INT, NULL));
  disp_elements.unbind();
  //GL_CHECK(glDisableVertexAttribArray(1));
  GL_CHECK(glDisableVertexAttribArray(0));
  disp_vertex.unbind();
  disp_shader.unbind();
  glPopMatrix();
}