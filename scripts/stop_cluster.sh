#!/bin/bash
# Stop Ray cluster
# Run this on both machines to fully stop the cluster

echo "Stopping Ray..."
ray stop --force

echo ""
echo "Ray cluster stopped."
echo ""
