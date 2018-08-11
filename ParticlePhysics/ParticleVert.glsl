#version 330

layout(location=0) in vec3 position;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

uniform sampler2D particlePos;
uniform sampler2D particleCol;

in int gl_InstanceID;
out vec4 col;

void main()
{
  vec4 pos = texelFetch(particlePos, ivec2(gl_InstanceID & 0xF, gl_InstanceID >> 4), 0);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position.xyz*pos.w + pos.xyz, 1.f);
  col = vec4(position.xyz, 1.f);
}