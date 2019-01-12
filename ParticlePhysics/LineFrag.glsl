#version 330

layout(location=0) out vec4 outputColor;

in vec4 col;

void main()
{
  outputColor = vec4(col);
}