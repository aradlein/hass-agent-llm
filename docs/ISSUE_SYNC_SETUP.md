# Issue Sync Setup Guide

This guide explains how to set up automatic issue synchronization from the public repository (`aradlein/hass-agent-llm`) to this private development repository (`aradlein/home-agent`).

## Overview

When an issue is created or updated in the public repository, a GitHub Action automatically creates or updates a corresponding issue in this private repository. This allows you to track and work on community-reported issues in your private development environment.

## Setup Instructions

### 1. Create a GitHub Personal Access Token

1. Go to GitHub Settings: https://github.com/settings/tokens
2. Click **"Generate new token"** → **"Generate new token (classic)"**
3. Give it a descriptive name: `Private Repo Issue Sync`
4. Set expiration (recommended: 90 days or 1 year)
5. Select the following scope:
   - ✅ **`repo`** (Full control of private repositories)
     - This is required to create and update issues in the private repo
6. Click **"Generate token"**
7. **IMPORTANT**: Copy the token immediately - you won't be able to see it again!

### 2. Add Token as Secret to Public Repository

1. Go to the **public** repository: https://github.com/aradlein/hass-agent-llm
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **"New repository secret"**
4. Name: `PRIVATE_REPO_TOKEN`
5. Value: Paste the token you created in step 1
6. Click **"Add secret"**

### 3. Copy the Workflow File to Public Repository

The workflow file is located at [.github/workflows/sync-issues-to-private.yml](.github/workflows/sync-issues-to-private.yml) in this repository.

You need to copy this file to the **public** repository (`hass-agent-llm`):

```bash
# In your local clone of hass-agent-llm
mkdir -p .github/workflows
cp path/to/home-agent/.github/workflows/sync-issues-to-private.yml .github/workflows/
git add .github/workflows/sync-issues-to-private.yml
git commit -m "Add automated issue sync to private dev repo"
git push
```

### 4. Test the Setup

1. Create a test issue in the public repository
2. Check the **Actions** tab in the public repository to see if the workflow ran successfully
3. Verify that the issue was created in this private repository with:
   - Same title and description
   - Label: `synced-from-public`
   - Link to the original public issue in the description

### 5. Troubleshooting

**Issue not syncing:**
- Check the Actions tab in the public repo for error messages
- Verify the `PRIVATE_REPO_TOKEN` secret is set correctly
- Ensure the token has the `repo` scope
- Check if the token has expired

**Permission errors:**
- The token must have access to the private repository
- If using a fine-grained token, ensure it has read/write access to Issues

**Duplicate issues:**
- The workflow searches for existing issues by the tracking comment
- If duplicates occur, check the search query in the workflow file

## How It Works

1. **Trigger**: When an issue is opened, edited, or labeled in the public repo
2. **Check**: The workflow checks if a corresponding issue already exists in the private repo
3. **Create/Update**:
   - If new: Creates a new issue with the same title, body, and labels (plus `synced-from-public`)
   - If existing: Updates the title, body, and labels
4. **Track**: Adds a note to the issue body linking back to the original public issue
5. **Confirm**: Adds a comment to the public issue confirming it was synced

## Sync Behavior

- **One-way sync**: Issues flow from public → private only
- **Synced properties**: Title, description, labels
- **Not synced**: Comments, assignees, milestones (can be added if needed)
- **Labels**: Original labels + `synced-from-public` label
- **Updates**: When a public issue is edited, the private issue is automatically updated

## Security Notes

- The workflow runs in the **public** repository but has access to the **private** repository via the token
- The token should have minimal required permissions (`repo` scope only)
- Regularly rotate the token for security
- Never commit the token to the repository - it should only exist as a secret

## Maintenance

### Renewing Expired Tokens

When your token expires:
1. Create a new token (follow step 1 above)
2. Update the `PRIVATE_REPO_TOKEN` secret in the public repo (step 2)
3. No code changes needed

### Disabling the Sync

To temporarily disable:
1. Go to the public repo → Actions → Workflows
2. Find "Sync Issues to Private Repo"
3. Click **"Disable workflow"**

To permanently remove:
1. Delete `.github/workflows/sync-issues-to-private.yml` from the public repo

## Customization

You can modify the workflow to:
- Sync only issues with specific labels
- Add assignees or milestones
- Sync comments from public to private
- Bi-directional sync (close private issue when public issue is closed)

See the [workflow file](.github/workflows/sync-issues-to-private.yml) for implementation details.
