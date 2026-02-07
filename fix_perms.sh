#!/bin/bash

# Some packages may lose their executable permissions during installation.
# This script restores the executable permissions for specific binaries.

chmod -R +x .venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/
chmod -R +x .venv/lib/python3.12/site-packages/lefthook/bin/lefthook-linux-x86_64/