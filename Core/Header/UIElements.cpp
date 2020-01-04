#include "UIElements.h"

#define TOGGLE_ANIMATION_SPEED    25.0f
#define BUTTON_ROUNDNESS_FRACTION 0.125f

#if ENV_APPLE

void ToggleButton(const char* buttonIdentifier, bool* value, int width, int height)
{
  ImVec2 position = ImGui::GetCursorScreenPos();

  // toggle value if button clicked
  ImGui::InvisibleButton(buttonIdentifier, ImVec2(width, height));
  if (ImGui::IsItemClicked())
  {
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
  ImGui::GetWindowDrawList()->AddRectFilled(position, ImVec2(position.x + width, position.y + height), backgroundColor, width * BUTTON_ROUNDNESS_FRACTION);
  position.x += sliderValue * (width - buttonWidth);
  ImGui::GetWindowDrawList()->AddRectFilled(position, ImVec2(position.x + buttonWidth, position.y + height), IM_COL32(255, 255, 255, 255), width * BUTTON_ROUNDNESS_FRACTION);

  ImGui::SameLine();
  ImGui::Text("%s", buttonIdentifier);
}

UIElement::UIElement(const string& identifier, const bool value, const char* iconFont, ushort iconId, const char* font)
{
  type = UI_ELEMENT_BOOL;
  this->identifier = identifier;
  boolValue = value;
  this->font = IOInterface::getFont(font);
}

UIElement::UIElement(const string& identifier, const char* value, const char* iconFont, ushort iconId, const char* font)
{
  type = UI_ELEMENT_STRING;
  this->identifier = identifier;
  if (IOInterface::checkImageExist(identifier.c_str()))
  {
    IOInterface::readImageFile(identifier.c_str(), &texture, 32, 32);
  }
  this->font = IOInterface::getFont(font);

  iconId = (iconId & 0xFFF) + IOInterface::getFontOffset(iconFont);
  stringValue.clear();
  const char iconCode[4] = {'\xEF', (char)(((iconId >> 6) & 0x3F) | 0x80), (char)((iconId & 0x3F) | 0x80), ' '};
  stringValue.append(iconCode, 4);
  stringValue.append(value);
}

void UIElement::render(uint width, uint height)
{
  if (type == UI_ELEMENT_BOOL)
  {
    ToggleButton(identifier.c_str(), &boolValue, width, height);
  }
  else if (type == UI_ELEMENT_STRING)
  {
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, width * BUTTON_ROUNDNESS_FRACTION);
    if (texture.get() != -1)
    {
      ImGui::ImageButton((ImTextureID)texture.get(), ImVec2(width, height), ImVec2(0, 0), ImVec2(1, 1), width * BUTTON_ROUNDNESS_FRACTION);
    }
    else
    {
      ImGui::Button(stringValue.c_str(), ImVec2(172, 24));
    }
    ImGui::PopStyleVar();
  }
}

#endif
