#include "UIElements.h"

#define TOGGLE_ANIMATION_SPEED  25.0f

#if ENV_APPLE

#include "imgui/imgui.h"
#include "imgui/imgui_impl_sdl.h"
#include "imgui/imgui_impl_opengl3.h"
#include "imgui/imgui_internal.h"
#include <SDL2/SDL.h>

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
  ImGui::GetWindowDrawList()->AddRectFilled(position, ImVec2(position.x + width, position.y + height), backgroundColor, width * 0.15f);
  position.x += sliderValue * (width - buttonWidth);
  ImGui::GetWindowDrawList()->AddRectFilled(position, ImVec2(position.x + buttonWidth, position.y + height), IM_COL32(255, 255, 255, 255), width * 0.15f);

  ImGui::SameLine();
  ImGui::Text("%s", buttonIdentifier);
}

void UIElement::render(uint width, uint height)
{
  if (type == UI_ELEMENT_BOOL)
  {
    ToggleButton(name.c_str(), &boolValue, width, height);
  }
}

#endif
