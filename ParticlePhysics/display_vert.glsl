#version 330

layout(location=0) in vec3 position;
//layout(location=1) in vec2 texCoord;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

out vec4 col;

void main()
{
  gl_Position = projectionMatrix*modelViewMatrix*vec4(position.xyz, 1.0);
  col = vec4(position.xyz,1);
}