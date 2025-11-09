# Icon Submission Guide

## Current Status

The Home Agent integration currently does **not have an icon displayed** in the Home Assistant UI. This is expected behavior for custom integrations.

## Why Icons Don't Show

Home Assistant **does not support local icons** for custom integrations. According to the official Home Assistant developer documentation:

- The `icon` field in `manifest.json` only works for built-in integrations
- Local icon files (icon.png, icon@2x.png, etc.) are ignored by the frontend
- Custom integrations must submit their icons to the official brands repository

## How to Add Icons

To display an icon in Home Assistant's UI, you need to submit the icon to the **Home Assistant Brands Repository**:

### Steps

1. Fork the repository: https://github.com/home-assistant/brands
2. Add your integration's icons following their structure:
   ```
   core_integrations/home_agent/
   ├── icon.png       (256x256 PNG)
   ├── icon@2x.png    (512x512 PNG)
   └── logo.png       (optional, for larger displays)
   ```
3. Submit a Pull Request

### Requirements

- Icons must be 256x256 pixels (standard) and optionally 512x512 (high-DPI)
- PNG format with transparent background
- Follow Home Assistant's branding guidelines
- The domain name must match your integration's domain (home_agent)

### Approval Process

- The Home Assistant team reviews submissions
- Once merged, icons will appear in the next Home Assistant release
- Icons are cached, so users may need to clear browser cache

## References

- [Official Blog Post: Logos for Custom Integrations](https://developers.home-assistant.io/blog/2020/05/08/logos-custom-integrations/)
- [Home Assistant Brands Repository](https://github.com/home-assistant/brands)
- [Brand Guidelines](https://github.com/home-assistant/brands/blob/master/GUIDELINES.md)

## Creating Icon Files for Submission

When you're ready to submit icons to the brands repository, you'll need to create:

- **icon.png** - 256x256 RGBA PNG with transparent background
- **icon@2x.png** - 512x512 RGBA PNG for high-DPI displays (optional but recommended)
- **logo.png** - Larger format for integration cards (optional)

These files should follow Home Assistant's branding guidelines and will need to be added to your fork of the brands repository before submitting a pull request.
