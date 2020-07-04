in lowp vec4 col;
in highp vec3 normal;

uniform highp float fillShader;
out lowp vec4 outputColor;

void main()
{
  lowp vec4 scaledColor = col + vec4(0.4f, 0.4f, 0.4f, 0.0f);
  if (fillShader == 1.f)
  {
    outputColor = scaledColor * clamp(dot(normal, vec3(0.f, 1.f, 0.f)), 0.2f, 1.f);
  }
  else
  {
    outputColor = vec4(0.75f, 0.75f, 0.75f, 1.0f);
  }
}
