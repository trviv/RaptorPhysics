in lowp vec4 col;
in highp vec3 normal;

out lowp vec4 outputColor;

void main()
{
  lowp vec4 scaledColor = col + vec4(0.4f, 0.4f, 0.4f, 1.0f);
  outputColor = scaledColor * clamp(dot(normal, vec3(0.f, 1.f, 0.f)), 0.2f, 1.f);
}
