#include "UIElements.h"

#define TOGGLE_ANIMATION_SPEED    25.0f
#define BUTTON_ROUNDNESS_FRACTION 0.125f

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
  if (IOInterface::checkImageExist(identifier.c_str()))
  {
    IOInterface::readImageFile(identifier.c_str(), &texture, 32, 32);
  }
  this->font = IOInterface::getFont(font);
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
    iconId = (iconId & 0xFFF) + IOInterface::getFontOffset(iconFont);
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
  ImGui::PushStyleVar(ImGuiStyleVar_GrabRounding, 32.f);

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
    ImGui::SliderInt(text.c_str(), &rangeValue, range[0], range[1]);
  }
  uiID = ImGui::GetItemID();
  this->size[0] = ImGui::GetItemRectSize()[0];
  this->size[1] = ImGui::GetItemRectSize()[1];

  this->pos[0]  = ImGui::GetCursorPos()[0];
  this->pos[1]  = ImGui::GetCursorPos()[1];

  ImGui::PopStyleVar(3);
  return changed;
}
