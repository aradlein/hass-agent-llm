# Issue #57: Submit to Home Assistant Brands

Tracking document for Home Assistant brands repository submission.

## Overview

- **Issue**: [#57 - Submit to Home Assistant Brands and design integration icon](https://github.com/aradlein/hass-agent-llm/issues/57)
- **Target Repository**: https://github.com/home-assistant/brands
- **Integration Domain**: `home_agent`
- **Public Repository**: `hass-agent-llm`

## Progress

### Completed

- [x] Researched Home Assistant brands repository requirements
- [x] Reviewed submission guidelines and file structure
- [x] Created AI image generation prompt for icon design

### In Progress

- [ ] Generate icon using AI image tools

### Pending

- [ ] Create icon.png (256x256) and icon@2x.png (512x512)
- [ ] Fork home-assistant/brands repository
- [ ] Create `custom_integrations/home_agent/` directory
- [ ] Submit PR with icon files

## Technical Requirements

### File Structure
```
custom_integrations/home_agent/
├── icon.png       (256x256 pixels)
└── icon@2x.png    (512x512 pixels)
```

### Icon Specifications
- **Format**: PNG with transparent background
- **Sizes**: 256x256 (icon.png) and 512x512 (icon@2x.png)
- **Max file size**: Under 50KB per file
- **Style**: Flat 2D vector, minimalist
- **Primary color**: Home Assistant blue (#03A9F4)
- **Must work**: At 32x32 pixels and on light/dark backgrounds
- **No text**: Icon should be purely graphical

## AI Image Generation Prompt

Use this prompt with AI image tools (DALL-E, Midjourney, Stable Diffusion, etc.):

```
I need to design an icon for a Home Assistant custom integration called "Home Agent". This is an LLM-powered conversational AI agent that provides intelligent home automation with persistent memory.

## Context
This icon will be submitted to the official Home Assistant brands repository (https://github.com/home-assistant/brands) for use in the Home Assistant UI. It needs to look professional and fit within the Home Assistant ecosystem while being distinct enough to not be confused with an official integration.

## Technical Requirements
- Create a square icon (1:1 aspect ratio)
- Provide at 512x512 pixels (I will create a 256x256 version from this)
- PNG format with transparent background
- File size should be optimized (under 50KB)
- Must be clearly recognizable when displayed at 32x32 pixels
- Must work on both light and dark backgrounds

## Design Requirements
- Minimalist, flat 2D vector style (no 3D, no photorealism)
- Use Home Assistant's blue color (#03A9F4) as the primary color
- Complementary colors: lighter blues, white accents are acceptable
- NO text or letters in the icon
- Simple geometric shapes
- Clean lines, no complex details or gradients

## Concept
The icon should visually communicate that this is an AI/LLM conversational agent for smart home control. Combine these elements:

1. **AI/Agent element** (choose one or combine):
   - Chat bubble / speech bubble
   - Neural network nodes
   - Friendly robot/assistant face
   - Stylized brain

2. **Home element** (subtle):
   - House silhouette or roofline
   - Home outline integrated into the design

## Style References
- Similar to other Home Assistant integration icons: simple, flat, monochromatic or limited color palette
- Think of icons like Slack, Discord, or other modern app icons - clean and recognizable

## What to Avoid
- Home Assistant's official logo or branding (would confuse users)
- Complex illustrations with many details
- Realistic or 3D rendered styles
- Dark or muted colors
- Any text, letters, or numbers
- Gradients or shadows

## Deliverable
A single clean icon design at 512x512 pixels, PNG format with transparent background, featuring a creative combination of AI/chat and home elements in Home Assistant blue (#03A9F4).
```

## PR Template for Brands Submission

When submitting to home-assistant/brands, use this PR description:

```markdown
## Summary
Adding brand assets for Home Agent, an advanced conversational AI agent integration.

## Integration Details
- **Domain**: `home_agent`
- **Type**: Custom Integration (HACS)
- **Repository**: https://github.com/aradlein/hass-agent-llm

## Assets Included
- [x] icon.png (256x256)
- [x] icon@2x.png (512x512)

## Integration Description
Home Agent is an LLM-powered conversational AI agent for Home Assistant that provides:
- Multi-provider LLM support (OpenAI, Anthropic, Google, Ollama)
- Persistent memory system for personalized interactions
- Streaming responses for real-time feedback
- Deep Home Assistant integration with entity control
```

## Resources

- [Home Assistant Brands Repository](https://github.com/home-assistant/brands)
- [Brands README](https://github.com/home-assistant/brands/blob/master/README.md)
- [Example: HACS brand files](https://github.com/home-assistant/brands/tree/master/custom_integrations/hacs)

## Next Steps for Continuation

1. **Generate icon** using the AI prompt above
2. **Resize** the generated image:
   - Save 512x512 version as `icon@2x.png`
   - Resize to 256x256 and save as `icon.png`
3. **Test** the icon at 32x32 to ensure readability
4. **Fork** the brands repository
5. **Create PR** with the icon files in `custom_integrations/home_agent/`
