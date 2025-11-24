/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#include "UIElements.h"
#include <Graphics/ImageLoader.h>
#include <Graphics/FontLoader.h>
#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>

#define TOGGLE_ANIMATION_SPEED    25.0f
#define BUTTON_ROUNDNESS_FRACTION 0.125f

static inline ImVec2 operator+(const ImVec2& lhs, const ImVec2& rhs)            { return ImVec2(lhs.x + rhs.x, lhs.y + rhs.y); }
static inline ImVec2 operator-(const ImVec2& lhs, const ImVec2& rhs)            { return ImVec2(lhs.x - rhs.x, lhs.y - rhs.y); }
static inline ImVec2 operator*(const ImVec2& lhs, const ImVec2& rhs)            { return ImVec2(lhs.x * rhs.x, lhs.y * rhs.y); }

inline bool CircularDial(const char* label, float* value, float rangeStart, float rangeEnd, bool isFloat, const int qualityFactor = 1)
{
  // settings
  const bool showPointer      = false;
  const bool showCircle       = false;
  const bool showArch         = true;
  const bool rotateDial       = false;
  const float triangleLength  = 0.5f;
  const float triangleWidth   = 0.0866025f;
  const float thicknessFactor   = 0.125f;
  const float outerRadiusFactor = 0.5f;
  const float rangeScale        = (rangeEnd - rangeStart);

  ImVec4 dialColor = ImVec4(255, 215, 0, 0);

  const int parts = qualityFactor+1;

  // variables
  bool valueChanged = false;
  float currentValue = *value; // always between [0 - 1]

  ImGuiContext& g = *GImGui;
  ImGuiWindow* window = ImGui::GetCurrentWindow();
  if (window->SkipItems)
    return false;

  ImDrawList* drawList = window->DrawList;
  ImGuiStyle& style = g.Style;

  g.NextItemData.ClearFlags();

  ImGui::BeginGroup();
  ImGui::PushID(label);

  // Setup
  const float barsWidth = ImGui::GetFrameHeight(); // Arbitrary smallish width of Hue/Alpha picking bars
  const float itemSize  = ImMax(barsWidth, ImGui::CalcItemWidth() - (barsWidth + style.ItemInnerSpacing.x)); // Saturation/Value picking box

  float oldValue = *value;

  const float wheelThickness    = itemSize * thicknessFactor;
  const float wheelOuterRadius  = itemSize * outerRadiusFactor;
  const float wheelInnerRadius  = wheelOuterRadius - wheelThickness;
  const float wheelRadius       = (wheelInnerRadius + wheelOuterRadius) * 0.5f;
  const ImVec2 wheelCenter(window->DC.CursorPos.x + (itemSize + barsWidth) * 0.5f, window->DC.CursorPos.y + itemSize * 0.5f);

  // Wheel logic to get current value
  ImGui::PushItemFlag(ImGuiItemFlags_NoNav, true);
  ImGui::InvisibleButton("hsv", ImVec2(itemSize + style.ItemInnerSpacing.x + barsWidth, itemSize));
  if (ImGui::IsItemActive())
  {
    const ImVec2 currentOffset  = ImGui::GetIO().MousePos - wheelCenter;
    const float initialDistSq   = ImLengthSqr(ImGui::GetIO().MouseClickedPos[0] - wheelCenter);
    if (initialDistSq >= mSqr(wheelInnerRadius - 1) && initialDistSq <= mSqr(wheelOuterRadius + 1))
    {
      // Interaction with the wheel
      currentValue = mATan2(currentOffset.y, currentOffset.x) / M_PI * 0.5f;
      if (currentValue < 0.0f)
      {
        currentValue += 1.0f;
      }

      // do not allow sudden jumps at ends
      if (currentValue < 0.5f && oldValue > .875f)
      {
        currentValue = 1.f;
      }
      else if (oldValue < 0.125f && currentValue > 0.5f)
      {
        currentValue = 0.f;
      }

      valueChanged = true;
    }
  }

  // this happens for some reason
  if (isnan(currentValue))
  {
    currentValue = rangeStart;
  }

  ImGui::PopItemFlag();

  const int styleAlpha8     = IM_F32_TO_INT8_SAT(style.Alpha);
  const ImU32 colorWhite    = IM_COL32(255, 255, 255, styleAlpha8);
  const ImU32 colorMidgrey  = IM_COL32(128, 128, 128, styleAlpha8);

  vector<ImU32> colorHues(parts+1);
  for (int n = 0; n <= parts; n++)
  {
    colorHues[n] = IM_COL32((uint)((dialColor.x * n)/parts), (uint)((dialColor.y * n)/parts), (uint)((dialColor.z * n)/parts), styleAlpha8);
  }

  // Render Hue Wheel
  const float aeps = 0.5f / wheelOuterRadius; // Half a pixel arc length in radians (2pi cancels out).
  const int segmentPerArc = mMax(qualityFactor * 4, (int)wheelOuterRadius / 12);

  float dialAngle = 0.f;
  if (rotateDial)
  {
    dialAngle = currentValue;
  }
  for (int n = 0; n < parts; n++)
  {
    const float a0 = (dialAngle + float(n)      / parts) * 2.0f * M_PI - aeps;
    const float a1 = (dialAngle + float(n+1.0f) / parts) * 2.0f * M_PI + aeps;
    const int vertexStartIndex = drawList->VtxBuffer.Size;
    drawList->PathArcTo(wheelCenter, wheelRadius, a0, a1, segmentPerArc);
    drawList->PathStroke(colorWhite, false, wheelThickness);
    const int vertexEndIndex = drawList->VtxBuffer.Size;

    // Paint colors over existing vertices
    ImVec2 gradientPoint0 = wheelCenter + ImVec2(mCos(a0), mSin(a0)) * ImVec2(wheelInnerRadius, wheelInnerRadius);
    ImVec2 gradientPoint1 = wheelCenter + ImVec2(mCos(a1), mSin(a1)) * ImVec2(wheelInnerRadius, wheelInnerRadius);
    ImGui::ShadeVertsLinearColorGradientKeepAlpha(drawList, vertexStartIndex, vertexEndIndex, gradientPoint0, gradientPoint1, colorHues[n], colorHues[n+1]);
  }

  // add text with current at the center
  char valueText[64] = "";
  if (isFloat)
  {
    sprintf(valueText, "%.3f", rangeStart + currentValue * rangeScale);
  }
  else
  {
    sprintf(valueText, "%d", (int)roundf(rangeStart + currentValue * rangeScale));
  }
  drawList->AddText(wheelCenter - ImGui::CalcTextSize(valueText) * ImVec2(0.5f, 0.5f), ImGui::ColorConvertFloat4ToU32(ImGui::GetStyleColorVec4(ImGuiCol_Text)), valueText);

  // draw a circle at the current position
  ImU32 colorUInt32 = ImGui::ColorConvertFloat4ToU32(ImVec4(1.f, 1.f, 1.f, style.Alpha));

  // Render Cursor + preview on Hue Wheel
  const float cosAngle = mCos(currentValue * 2.0f * M_PI);
  const float sinAngle = mSin(currentValue * 2.0f * M_PI);

  if (showCircle)
  {
    const ImVec2 cursorPosition = wheelCenter + ImVec2(cosAngle, sinAngle) * ImVec2(wheelRadius, wheelRadius);
    float cursorRadius = wheelThickness * (valueChanged ? 0.85f : 0.55f);
    int cursorSegments = mCrop((int)(cursorRadius / 1.4f), qualityFactor*4, qualityFactor*8);

    drawList->AddCircle(cursorPosition, cursorRadius, colorWhite, cursorSegments, qualityFactor);
  }

  if (showPointer)
  {
    // Note: the triangle is displayed rotated with triangle_pa pointing to Hue, but most coordinates stays unrotated for logic.
    float triangleRadius  = wheelInnerRadius - (int)(itemSize * 0.027f);
    ImVec2 trianglePointA = ImVec2(triangleRadius, 0.0f); // Hue point.
    ImVec2 trianglePointB = ImVec2(triangleRadius * triangleLength, triangleRadius * -triangleWidth); // Black point.
    ImVec2 trianglePointC = ImVec2(triangleRadius * triangleLength, triangleRadius * triangleWidth); // White point.

    // Render SV triangle (rotated according to hue)
    const ImVec2 triangleA = wheelCenter + ImRotate(trianglePointA, cosAngle, sinAngle);
    const ImVec2 triangleB = wheelCenter + ImRotate(trianglePointB, cosAngle, sinAngle);
    const ImVec2 triangleC = wheelCenter + ImRotate(trianglePointC, cosAngle, sinAngle);
    const ImVec2 uv_white = ImGui::GetFontTexUvWhitePixel();
    drawList->PrimReserve(6, 6);
    drawList->PrimVtx(triangleA, uv_white, colorUInt32);
    drawList->PrimVtx(triangleB, uv_white, colorUInt32);
    drawList->PrimVtx(triangleC, uv_white, colorWhite);
    drawList->PrimVtx(triangleA, uv_white, 0);
    drawList->PrimVtx(triangleB, uv_white, IM_COL32(0, 0, 0, styleAlpha8));
    drawList->PrimVtx(triangleC, uv_white, 0);
    drawList->AddTriangle(triangleA, triangleB, triangleC, colorMidgrey, 1.5f);
  }

  if (showArch)
  {
    int subParts = parts * 4;// * 4 + ((parts&1)?1:0);
    for (int n = 0; n < 1; n++)
    {
      const float a0 = (currentValue + float(n-1.0f) * 0.5f / subParts) * 2.0f * M_PI - aeps;
      const float a1 = (currentValue + float(n+1.0f) * 0.5f / subParts) * 2.0f * M_PI + aeps;

      const ImU32 colorWhite    = IM_COL32(255, 255, 255, 64);

      drawList->PathArcTo(wheelCenter, wheelRadius, a0, a1, segmentPerArc/2);
      drawList->PathStroke(colorWhite, false, wheelThickness * 1.5f);
    }
  }

  ImGui::PopID();
  ImGui::EndGroup();

  *value = currentValue;

  if (valueChanged && oldValue == *value)
  {
    valueChanged = false;
  }
  if (valueChanged)
  {
    ImGui::MarkItemEdited(window->DC.LastItemId);
  }

  return valueChanged;
}

template<typename type> bool CircularDial(const char* label, type* value, type rangeStart, type rangeEnd, const int qualityFactor = 1)
{
  float floatValue;
  const type rangeScale = (rangeEnd - rangeStart);
  floatValue = ((float)(*value - rangeStart))/rangeScale;

  bool ret = CircularDial(label, &floatValue, rangeStart, rangeEnd, std::is_same<float, type>::value, qualityFactor);
  if (!std::is_same<float, type>::value)
  {
    *value = rangeStart + (type)roundf(floatValue * rangeScale);
  }
  else
  {
    *value = rangeStart + floatValue * rangeScale;
  }
  return ret;
}

void ToggleButton(const char* buttonIdentifier, const char* text, bool* value, int width, int height, bool& changed)
{
  if (!*value)
  {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f, 1.f, 1.f, 1.f));
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.f, 0.f, 0.f, 0.f));
  }
  else
  {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.f, 0.f, 0.f, 1.f));
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.25999999f, 0.980000019f, 0.589999974f, 1.f));
  }
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.25999999f, 0.980000019f, 0.589999974f, 0.400000006f));
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.0599999987f, 0.980000019f, 0.529999971f, 1.f));

  if (ImGui::ButtonEx(text, ImVec2(width, height), ImGuiButtonFlags_PressedOnClick))
  {
    changed = true;
    *value = !*value;
  }

  ImGui::PopStyleColor(4);
}

void ToggleButton2(const char* buttonIdentifier, bool* value, int width, int height, bool& changed)
{
  ImVec2 position = ImGui::GetCursorScreenPos();

  // toggle value if button clicked
  ImGui::InvisibleButton(buttonIdentifier, ImVec2(width, height));
  if (ImGui::IsItemClicked())
  {
    changed = true;
    *value = !*value;
  }

  // set slider position based on button's value or if its currently animating
  float sliderValue = *value ? 1.0f : 0.0f;
  if (ImGui::GetCurrentContext()->LastActiveId == ImGui::GetCurrentContext()->CurrentWindow->GetID(buttonIdentifier))
  {
    const float animationTime = ImSaturate(ImGui::GetCurrentContext()->LastActiveIdTimer * TOGGLE_ANIMATION_SPEED);
    sliderValue = *value ? animationTime : (1.0f - animationTime);
  }

  // set hover color
  ImU32 backgroundColor;
  if (ImGui::IsItemHovered())
  {
    backgroundColor = ImGui::GetColorU32(ImLerp(ImVec4(0.78f, 0.78f, 0.78f, 1.0f), ImVec4(0.64f, 0.83f, 0.34f, 1.0f), sliderValue));
  }
  else
  {
    backgroundColor = ImGui::GetColorU32(ImLerp(ImVec4(0.85f, 0.85f, 0.85f, 1.0f), ImVec4(0.56f, 0.83f, 0.26f, 1.0f), sliderValue));
  }

  const float buttonWidth = width * 0.5f;
  ImGui::GetWindowDrawList()->AddRectFilled(position, ImVec2(position.x + width, position.y + height), backgroundColor, height * BUTTON_ROUNDNESS_FRACTION);
  position.x += sliderValue * (width - buttonWidth);
  ImGui::GetWindowDrawList()->AddRectFilled(position, ImVec2(position.x + buttonWidth, position.y + height), IM_COL32(255, 255, 255, 255), height * BUTTON_ROUNDNESS_FRACTION);

  ImGui::SameLine();
  ImGui::Text("%s", buttonIdentifier);
}

float UIElement::ButtonWidth = 160;
float UIElement::ButtonHeight = 32;

UIElement::UIElement(const string& identifier, const char* iconFont, ushort iconId, const char* font)
{
  alignment = UIObject::FloatX|UIObject::FloatY;
  this->identifier = identifier;
  if (ImageLoader::checkImageExist(identifier.c_str()))
  {
    ImageLoader::readImageFile(identifier.c_str(), &texture, 32, 32);
  }
  this->font = FontLoader::getFont(font);
  this->autoWidth = false;
}

UIElement::UIElement(const string& identifier, const bool value, const char* iconFont, ushort iconId, const char* font)
  :UIElement(identifier, iconFont, iconId, font)
{
  type = UIElementType::Bool;
  this->text = getIconAsString(iconFont, iconId) + identifier;
  boolValue = value;
}

UIElement::UIElement(const string& identifier, const string& text, const char* iconFont, ushort iconId, const char* font)
  :UIElement(identifier, iconFont, iconId, font)
{
  type = UIElementType::String;
  this->text = getIconAsString(iconFont, iconId) + text;
}

UIElement::UIElement(const string& identifier, const int min, const int max, const char* iconFont, ushort iconId, const char* font)
  :UIElement(identifier, iconFont, iconId, font)
{
  type = UIElementType::Slider;
  this->text = getIconAsString(iconFont, iconId);
  range[0] = min;
  range[1] = max;
  rangeValue = 0;
}

UIElement::~UIElement()
{
}

string UIElement::getIconAsString(const char* iconFont, ushort iconId)
{
  string str;
  if (iconId)
  {
    iconId = (iconId & 0xFFF) + FontLoader::getFontOffset(iconFont);
    const char iconCode[4] = {'\xEF', (char)(((iconId >> 6) & 0x3F) | 0x80), (char)((iconId & 0x3F) | 0x80), ' '};
    str.append(iconCode, 4);
  }

  return str;
}

bool UIElement::render()
{
  bool changed = false;
  ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0.0f, 0.5f));
  ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, ButtonHeight * BUTTON_ROUNDNESS_FRACTION);
  ImGui::PushStyleVar(ImGuiStyleVar_GrabRounding, 24.f * BUTTON_ROUNDNESS_FRACTION);

  ImVec2 size(UIElement::ButtonWidth, UIElement::ButtonHeight);
  if (autoWidth)
  {
    size.x = 0;
  }

  if (type == UIElementType::Bool)
  {
    ToggleButton(identifier.c_str(), text.c_str(), &boolValue, size.x, size.y, changed);
  }
  else if (type == UIElementType::String)
  {
    if (texture.get() != -1)
    {
      ImGui::ImageButton((ImTextureID)texture.get(), ImVec2(0, 0), ImVec2(0, 0), ImVec2(1, 1), 0 * BUTTON_ROUNDNESS_FRACTION);
    }
    else
    {
      ImGui::ButtonEx(text.c_str(), size, ImGuiButtonFlags_PressedOnClick);
    }
  }
  else if (type == UIElementType::Slider)
  {
    ImGui::Text("");
    ImGui::SliderInt(identifier.c_str(), &rangeValue, range[0], range[1]);
//    CircularDial(identifier.c_str(), &rangeValue, range[0], range[1], 4);
    ImGui::Text("");
  }
  uiID = ImGui::GetItemID();
  this->size[0] = ImGui::GetItemRectSize()[0];
  this->size[1] = ImGui::GetItemRectSize()[1];

  ImGui::PopStyleVar(3);
  return changed;
}
