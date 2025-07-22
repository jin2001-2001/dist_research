#!/bin/bash

PORT=29500

# Search for and kill the processes occupying the specified port
PID=$(lsof -t -i:$PORT)
if [ -n "$PID" ]; then
  echo "Killing process on port $PORT with PID: $PID"
  kill -9 $PID
else
  echo "No process is using port $PORT"
fi

echo "Killing all torchrun processes..."
pkill -9 -f torchrun || echo "No torchrun process found."

echo "Killing all python processes..."
pkill -9 -f python || echo "No python process found."

echo "Killing all gloo-related processes..."
pgrep -f gloo | xargs -r kill -9 || echo "No gloo process found."

echo "Killing all nccl-related processes..."
pgrep -f nccl | xargs -r kill -9 || echo "No nccl process found."

echo "Done."
