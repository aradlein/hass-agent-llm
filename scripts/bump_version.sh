#!/bin/bash
# Version bumping script for Home Agent
# Usage: ./scripts/bump_version.sh [major|minor|patch|beta|alpha|rc] [message]

set -e

MANIFEST_FILE="custom_components/home_agent/manifest.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    print_error "jq is required but not installed. Install with: apt-get install jq"
    exit 1
fi

# Get current version
CURRENT_VERSION=$(jq -r '.version' "$MANIFEST_FILE")
print_info "Current version: $CURRENT_VERSION"

# Parse current version
if [[ $CURRENT_VERSION =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)(-([a-zA-Z0-9]+))?$ ]]; then
    MAJOR="${BASH_REMATCH[1]}"
    MINOR="${BASH_REMATCH[2]}"
    PATCH="${BASH_REMATCH[3]}"
    SUFFIX="${BASH_REMATCH[5]}"
else
    print_error "Invalid version format: $CURRENT_VERSION"
    exit 1
fi

# Determine bump type
BUMP_TYPE="${1:-patch}"
COMMIT_MESSAGE="${2:-Bump version to}"

case "$BUMP_TYPE" in
    major)
        MAJOR=$((MAJOR + 1))
        MINOR=0
        PATCH=0
        SUFFIX=""
        ;;
    minor)
        MINOR=$((MINOR + 1))
        PATCH=0
        SUFFIX=""
        ;;
    patch)
        PATCH=$((PATCH + 1))
        SUFFIX=""
        ;;
    beta)
        if [[ -z "$SUFFIX" ]]; then
            SUFFIX="beta"
        else
            print_warning "Version already has suffix: $SUFFIX"
            print_info "Keeping current suffix and bumping patch"
            PATCH=$((PATCH + 1))
        fi
        ;;
    alpha)
        SUFFIX="alpha"
        ;;
    rc)
        SUFFIX="rc"
        ;;
    release)
        # Remove suffix for release
        SUFFIX=""
        ;;
    *)
        print_error "Invalid bump type: $BUMP_TYPE"
        echo "Usage: $0 [major|minor|patch|beta|alpha|rc|release] [commit_message]"
        exit 1
        ;;
esac

# Construct new version
if [[ -n "$SUFFIX" ]]; then
    NEW_VERSION="$MAJOR.$MINOR.$PATCH-$SUFFIX"
else
    NEW_VERSION="$MAJOR.$MINOR.$PATCH"
fi

print_info "New version: $NEW_VERSION"

# Confirm with user
read -p "Bump version from $CURRENT_VERSION to $NEW_VERSION? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_warning "Version bump cancelled"
    exit 0
fi

# Update manifest.json
jq --arg version "$NEW_VERSION" '.version = $version' "$MANIFEST_FILE" > "${MANIFEST_FILE}.tmp"
mv "${MANIFEST_FILE}.tmp" "$MANIFEST_FILE"

print_success "Updated $MANIFEST_FILE to version $NEW_VERSION"

# Git operations
if git rev-parse --git-dir > /dev/null 2>&1; then
    print_info "Staging changes..."
    git add "$MANIFEST_FILE"

    print_info "Creating commit..."
    git commit -m "$COMMIT_MESSAGE $NEW_VERSION"

    print_success "Committed version bump"

    # Ask if user wants to create a tag
    read -p "Create git tag v$NEW_VERSION? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git tag -a "v$NEW_VERSION" -m "Release version $NEW_VERSION"
        print_success "Created tag v$NEW_VERSION"

        read -p "Push commit and tag to origin? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git push && git push --tags
            print_success "Pushed to origin"
        else
            print_warning "Remember to push: git push && git push --tags"
        fi
    else
        print_warning "Remember to create and push tag: git tag -a v$NEW_VERSION -m 'Release version $NEW_VERSION' && git push --tags"
    fi
else
    print_warning "Not a git repository - skipping git operations"
fi

print_success "Version bump complete!"
echo
print_info "Next steps:"
echo "  1. Review the changes"
echo "  2. Push to GitHub: git push && git push --tags"
echo "  3. GitHub Actions will automatically create a release"
