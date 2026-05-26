# Deploying Armdroid on NVIDIA Jetson

This guide covers building and running the armdroid Docker image on
NVIDIA Jetson platforms (Orin Nano, Orin NX, AGX Orin).

---

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| JetPack SDK | 6.0 (L4T r36.3) | 6.1 (L4T r36.4) |
| CUDA | 12.2 | 12.6 |
| Python | 3.11 | 3.11 |
| Docker | 24.0 | 27.x |
| NVIDIA Container Toolkit | 1.14 | 1.16 |
| RAM | 8 GB | 16 GB+ |

### Install NVIDIA Container Toolkit

```bash
# Add the NVIDIA apt repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## Building the Docker Image

### Quick build (default JetPack 6.1)

```bash
cd /path/to/armdroid
docker compose -f docker-compose.jetson.yml build
```

### Custom JetPack version

```bash
docker build \
  --build-arg L4T_TAG=r36.3.0-py3 \
  --build-arg CUDA_ARCH_LIST="8.7" \
  -t armdroid:jetson .
```

### Build arguments

| ARG | Default | Description |
|-----|---------|-------------|
| `L4T_TAG` | `r36.4.0-py3` | L4T PyTorch base image tag |
| `PYTHON_VERSION` | `3.11` | Python version for build |
| `CUDA_ARCH_LIST` | `8.7` | CUDA compute capability (Orin = 8.7) |
| `APP_USER` | `armdroid` | Non-root user inside container |
| `APP_UID` | `1000` | UID for the app user |

---

## Running

### With Docker Compose (recommended)

```bash
# Set your HMAC key (required for authenticated transports)
export ARMDROID_HMAC_KEY="your-secret-key-here"

# Start the container
docker compose -f docker-compose.jetson.yml up -d

# View logs
docker compose -f docker-compose.jetson.yml logs -f

# Stop
docker compose -f docker-compose.jetson.yml down
```

### Direct `docker run`

```bash
docker run -d \
  --name armdroid \
  --runtime nvidia \
  --device /dev/ttyUSB0 \
  -v $(pwd)/config:/app/config:ro \
  -v $(pwd)/.secrets:/app/.secrets:ro \
  -e ARMDROID_HMAC_KEY \
  armdroid:jetson \
  --config /app/config/tower_of_hanoi.yaml --mode run
```

---

## Device Passthrough

### ESP32 Serial

The ESP32 connects via USB serial. Pass it through with `--device`:

```bash
# Find the ESP32 device path
ls /dev/ttyUSB*

# Typical: /dev/ttyUSB0
docker run --device /dev/ttyUSB0 ...
```

> **Tip:** If the device path changes on reconnect, use a udev rule
> to create a stable symlink:
>
> ```bash
> # /etc/udev/rules.d/99-armdroid-esp32.rules
> SUBSYSTEM=="tty", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", \
>   SYMLINK+="armdroid-esp32", MODE="0666"
> ```

### RealSense D435i

```bash
docker run \
  --device /dev/video0 \
  --device /dev/video1 \
  -v /dev/bus/usb:/dev/bus/usb \
  ...
```

### BLE Transport

BLE requires host network mode and access to the Bluetooth adapter:

```bash
docker run \
  --network host \
  --privileged \
  -v /var/run/dbus:/var/run/dbus \
  ...
```

---

## Configuration

### YAML Overlays

Mount your configuration at `/app/config/`:

```bash
docker run -v $(pwd)/config:/app/config:ro ...
```

The container reads `config/tower_of_hanoi.yaml` by default. Override
with the `--config` flag:

```bash
docker run ... armdroid:jetson \
  --config /app/config/my_custom_config.yaml
```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ARMDROID_HMAC_KEY` | HMAC-SHA256 key for authenticated transports | Yes (if auth enabled) |
| `ARMDROID_LOG_LEVEL` | Log level (DEBUG, INFO, WARNING, ERROR) | No (default: INFO) |
| `ESP32_DEVICE` | Host device path for ESP32 | No (default: /dev/ttyUSB0) |
| `L4T_TAG` | L4T base image tag (build-time only) | No |

---

## Health Check

The container includes a built-in health check that runs every 30 seconds:

```bash
# Check container health status
docker inspect --format='{{.State.Health.Status}}' armdroid

# Run the health check manually inside the container
docker exec armdroid python /app/scripts/jetson_health_check.py

# JSON output
docker exec armdroid python /app/scripts/jetson_health_check.py --json
```

---

## Troubleshooting

### GPU not detected

```
[FAIL] gpu: CUDA not available
```

- Verify NVIDIA Container Toolkit: `nvidia-ctk --version`
- Check Docker runtime: `docker info | grep -i runtime`
- Verify GPU inside container: `docker run --runtime nvidia --rm nvcr.io/nvidia/l4t-pytorch:r36.4.0-py3 nvidia-smi`

### Serial permission denied

```
PermissionError: [Errno 13] Permission denied: '/dev/ttyUSB0'
```

- The container user is added to the `dialout` group, but the host
  device permissions must allow it. Fix with:

```bash
sudo chmod 666 /dev/ttyUSB0
# Or add a udev rule (see Device Passthrough above)
```

### RealSense not detected

- Ensure `librealsense2` is installed in the image (it is by default)
- Pass USB bus: `-v /dev/bus/usb:/dev/bus/usb`
- Check with: `docker exec armdroid realsense-viewer` (if display available)

### Out of memory

- Reduce YOLO model size: use `yolo11n` instead of `yolo11s`
- Reduce replay buffer in training config
- Monitor with: `docker stats armdroid`
