#version 150

in vec3 position;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

uniform sampler2D boundingBoxes;

out vec4 col;

void main()
{
  vec4 min = texelFetch(boundingBoxes, ivec2((gl_InstanceID*2) & 0x7F, (gl_InstanceID*2) >> 7), 0);
  vec4 max = texelFetch(boundingBoxes, ivec2((gl_InstanceID*2 + 1) & 0x7F, (gl_InstanceID*2 + 1) >> 7), 0);

  vec4 pos = projectionMatrix * modelViewMatrix * vec4(((max.xyz + min.xyz) + position * (max.xyz - min.xyz)) * 0.5f, 1.f);

  gl_Position = pos;
  col = vec4(0.0f, 0.f, 0.f, 0.50f);
}
