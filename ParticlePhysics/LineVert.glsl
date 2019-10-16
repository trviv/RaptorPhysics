uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

uniform sampler2D particlePos;
uniform sampler2D particleCollData;

out vec4 col;

#define texureWidth     128
#define texureWidthExp  7

void main()
{
  vec4 pos = texelFetch(particlePos, ivec2(gl_InstanceID & (texureWidth - 1), gl_InstanceID >> texureWidthExp), 0);

  if (gl_VertexID>0)
  {
    vec4 gradient = texelFetch(particleCollData, ivec2((gl_InstanceID * 2 + 1) & (texureWidth - 1), (gl_InstanceID * 2) >> texureWidthExp), 0);
    pos += vec4(gradient.xyz, 1.0f);
  }

  pos = projectionMatrix * modelViewMatrix * vec4(pos.xyz, 1.f);

  gl_Position = pos;
  col = vec4(1.f, 0.f, 0.f, 1.f);
}
