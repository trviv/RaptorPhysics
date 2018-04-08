#version 330

layout(location=0) in vec3 position;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

out vec4 col;

void main()
{
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position.xyz, 1.f);
  col = vec4(position.xyz, 1.f);
}