in highp vec4 col;
in highp vec3 normal;

uniform int fill;
out highp vec4 outputColor;

void main()
{
  highp vec4 scaledColor = col + vec4(0.4f, 0.4f, 0.4f, 1.0f);
  if (fill == 1)
  {
    outputColor = scaledColor * clamp(dot(normal, vec3(0.f, 1.f, 0.f)), 0.2f, 1.f);
  }
  else
  {
    outputColor = vec4(0.0f, 0.0f, 0.0f, 1.0f);
  }
}
