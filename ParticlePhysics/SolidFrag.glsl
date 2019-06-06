#version 150

in vec4 col;
out vec4 outputColor;

void main()
{
  //outputColor = vec4(0.0,1.0,0.0,1.0);//vec4((col+1.0)/2.0);
  outputColor = vec4((col+1.0)/2.0);
}
