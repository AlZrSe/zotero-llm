#!/bin/sh

# Wait for Qdrant to be ready
echo "Waiting for Qdrant to be ready..."
until curl -s -f "http://${QDRANT_HOST}:${QDRANT_PORT}/healthz" > /dev/null; do
    echo "Waiting for Qdrant..."
    sleep 5
done
echo "Qdrant is ready!"

# Execute the command passed to docker run or docker-compose
exec "$@"
