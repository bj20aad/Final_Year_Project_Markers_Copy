Setup Docker:

docker run -it --rm \
  --memory="3.8g" \
  --cpus="4.0" \
  -v "$(pwd):/app" \
  -w /app \
  python:3.9-slim