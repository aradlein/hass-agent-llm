#!/bin/bash
# Quick wrapper script for updating Home Agent
# Can be copied to /usr/local/bin for easy access from anywhere

SCRIPT_DIR="/root/home-agent/scripts"
UPDATE_SCRIPT="${SCRIPT_DIR}/update_integration.sh"

# Check if update script exists
if [ ! -f "$UPDATE_SCRIPT" ]; then
    echo "Error: Update script not found at $UPDATE_SCRIPT"
    echo "Please ensure the repository is cloned to /root/home-agent"
    exit 1
fi

# Forward all arguments to the update script
exec "$UPDATE_SCRIPT" "$@"
