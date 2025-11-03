#!/bin/bash

# Home Agent Integration Update Script
# Updates the Home Assistant integration from git repository to production
# Usage: ./update_integration.sh [OPTIONS] [TAG]
#
# Options:
#   --dry-run    Show commands without executing them
#   --latest     Use the latest tag (default if no tag specified)
#   --help       Show this help message
#
# Examples:
#   ./update_integration.sh --dry-run v0.4.4-beta
#   ./update_integration.sh --latest
#   ./update_integration.sh v0.4.3-beta

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
GIT_REPO_PATH="/root/home-agent"
HA_INTEGRATION_PATH="/root/config/custom_components/home_agent"
INTEGRATION_SOURCE_PATH="${GIT_REPO_PATH}/custom_components/home_agent"

# Default values
DRY_RUN=false
USE_LATEST=false
TARGET_TAG=""

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_command() {
    echo -e "${YELLOW}[CMD]${NC} $1"
}

# Function to execute or print command based on dry-run mode
execute_cmd() {
    local cmd="$1"
    local description="$2"

    if [ -n "$description" ]; then
        print_info "$description"
    fi

    print_command "$cmd"

    if [ "$DRY_RUN" = false ]; then
        eval "$cmd"
        return $?
    fi
    return 0
}

# Function to show help
show_help() {
    cat << EOF
Home Agent Integration Update Script

Updates the Home Assistant integration from git repository to production.

USAGE:
    $0 [OPTIONS] [TAG]

OPTIONS:
    --dry-run    Show all commands without executing them
    --latest     Use the latest git tag (default if no tag specified)
    --help       Show this help message

ARGUMENTS:
    TAG          Specific git tag to checkout (e.g., v0.4.4-beta)
                 If not provided, uses --latest

PATHS:
    Git Repository:    ${GIT_REPO_PATH}
    Source:           ${INTEGRATION_SOURCE_PATH}
    Destination:      ${HA_INTEGRATION_PATH}

EXAMPLES:
    # Dry run with specific tag
    $0 --dry-run v0.4.4-beta

    # Update to latest tag
    $0 --latest

    # Update to specific tag
    $0 v0.4.3-beta

    # Dry run to see what the latest tag would do
    $0 --dry-run --latest

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --latest)
            USE_LATEST=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        -*)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            TARGET_TAG="$1"
            shift
            ;;
    esac
done

# Banner
echo "========================================"
echo "Home Agent Integration Update Script"
echo "========================================"
echo ""

if [ "$DRY_RUN" = true ]; then
    print_warning "DRY RUN MODE - No changes will be made"
    echo ""
fi

# Validation: Check if git repo exists
if [ ! -d "$GIT_REPO_PATH" ]; then
    print_error "Git repository not found at: $GIT_REPO_PATH"
    exit 1
fi

if [ ! -d "$GIT_REPO_PATH/.git" ]; then
    print_error "Not a git repository: $GIT_REPO_PATH"
    exit 1
fi

print_success "Git repository found at: $GIT_REPO_PATH"

# Change to git repository
execute_cmd "cd $GIT_REPO_PATH" "Changing to git repository"

# Fetch latest tags
execute_cmd "git fetch --tags" "Fetching latest tags from remote"

# Determine target tag
if [ "$USE_LATEST" = true ] || [ -z "$TARGET_TAG" ]; then
    if [ "$DRY_RUN" = false ]; then
        TARGET_TAG=$(git describe --tags --abbrev=0)
    else
        # In dry run, we still need to get the latest tag
        TARGET_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "latest-tag-placeholder")
    fi
    print_info "Using latest tag: $TARGET_TAG"
else
    print_info "Using specified tag: $TARGET_TAG"
fi

# Verify tag exists
if [ "$DRY_RUN" = false ]; then
    if ! git rev-parse "$TARGET_TAG" >/dev/null 2>&1; then
        print_error "Tag '$TARGET_TAG' does not exist"
        echo ""
        print_info "Available tags:"
        git tag | tail -10
        exit 1
    fi
fi

# Show current status
print_info "Current git status:"
if [ "$DRY_RUN" = false ]; then
    git status --short
fi

# Stash any local changes
execute_cmd "git stash" "Stashing any local changes"

# Checkout the target tag
execute_cmd "git checkout $TARGET_TAG" "Checking out tag: $TARGET_TAG"

# Verify source directory exists
if [ ! -d "$INTEGRATION_SOURCE_PATH" ]; then
    print_error "Integration source not found at: $INTEGRATION_SOURCE_PATH"
    exit 1
fi

print_success "Integration source found at: $INTEGRATION_SOURCE_PATH"

# Check if destination directory exists
if [ ! -d "$HA_INTEGRATION_PATH" ]; then
    print_warning "Destination directory does not exist: $HA_INTEGRATION_PATH"
    execute_cmd "mkdir -p $HA_INTEGRATION_PATH" "Creating destination directory"
fi

# Backup existing installation (if not dry-run)
BACKUP_PATH="${HA_INTEGRATION_PATH}_backup_$(date +%Y%m%d_%H%M%S)"
if [ -d "$HA_INTEGRATION_PATH" ] && [ "$(ls -A $HA_INTEGRATION_PATH 2>/dev/null)" ]; then
    execute_cmd "cp -r $HA_INTEGRATION_PATH $BACKUP_PATH" "Creating backup at: $BACKUP_PATH"
else
    print_info "No existing installation to backup"
fi

# Remove old installation
if [ -d "$HA_INTEGRATION_PATH" ]; then
    execute_cmd "rm -rf ${HA_INTEGRATION_PATH}/*" "Removing old installation files"
fi

# Copy new version
execute_cmd "cp -r ${INTEGRATION_SOURCE_PATH}/* ${HA_INTEGRATION_PATH}/" "Copying new integration files"

# Verify installation
print_info "Verifying installation..."
if [ "$DRY_RUN" = false ]; then
    if [ -f "${HA_INTEGRATION_PATH}/manifest.json" ]; then
        print_success "Installation verified - manifest.json found"
        echo ""
        print_info "Installed version information:"
        grep -E '"version"|"name"' "${HA_INTEGRATION_PATH}/manifest.json" || true
    else
        print_error "Installation verification failed - manifest.json not found"
        exit 1
    fi
else
    print_command "test -f ${HA_INTEGRATION_PATH}/manifest.json"
    print_command "grep -E '\"version\"|\"name\"' ${HA_INTEGRATION_PATH}/manifest.json"
fi

# Summary
echo ""
echo "========================================"
echo "Update Summary"
echo "========================================"
echo "Git Tag:        $TARGET_TAG"
echo "Source:         $INTEGRATION_SOURCE_PATH"
echo "Destination:    $HA_INTEGRATION_PATH"
if [ "$DRY_RUN" = false ] && [ -n "$BACKUP_PATH" ]; then
    echo "Backup:         $BACKUP_PATH"
fi
echo "========================================"
echo ""

if [ "$DRY_RUN" = true ]; then
    print_warning "DRY RUN COMPLETED - No changes were made"
    echo ""
    print_info "To perform the actual update, run without --dry-run flag:"
    echo "  $0 $TARGET_TAG"
else
    print_success "Update completed successfully!"
    echo ""
    print_warning "IMPORTANT: Restart Home Assistant for changes to take effect"
    echo ""
    print_info "To restart Home Assistant, run:"
    echo "  ha core restart"
    echo ""
    print_info "If you encounter issues, restore from backup:"
    echo "  rm -rf $HA_INTEGRATION_PATH"
    echo "  mv $BACKUP_PATH $HA_INTEGRATION_PATH"
    echo "  ha core restart"
fi

echo ""
