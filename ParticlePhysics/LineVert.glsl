uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

uniform sampler2D particlePos;
uniform sampler2D particleSDFGrad;

out vec4 col;

void main()
{
  vec4 pos = texelFetch(particlePos, ivec2(gl_InstanceID & 0xF, gl_InstanceID >> 4), 0);

  if (gl_VertexID>0)
  {
    vec4 gradient = texelFetch(particleSDFGrad, ivec2(gl_InstanceID & 0xF, gl_InstanceID >> 4), 0);
    pos += vec4(gradient.xyz, 1.0f);
  }

  pos = projectionMatrix * modelViewMatrix * vec4(pos.xyz, 1.f);

  gl_Position = pos;
  col = vec4(1.f, 0.f, 0.f, 1.f);
}
