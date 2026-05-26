# =============================================================================
# Armdroid — Multi-stage Dockerfile for NVIDIA Jetson (L4T) deployment
#
# Targets NVIDIA Jetson Orin (JetPack 6.x / L4T r36.x) with CUDA + PyTorch.
# Also builds on x86_64 for CI smoke-testing (use the x86 stage override).
#
# Build:
#   docker build -t armdroid:latest .
#   docker build --build-arg L4T_TAG=r36.4.0-py3 -t armdroid:jetson .
#
# Run (Jetson with ESP32 + RealSense):
#   docker run --runtime nvidia --device /dev/ttyUSB0 --device /dev/video0 \
#     -v $(pwd)/config:/app/config:rw \
#     -e ARMDROID_HMAC_KEY \
#     armdroid:jetson
# =============================================================================

# ---------------------------------------------------------------------------
# Build arguments — NO hardcoded versions (AGENTS.md Rule 1)
# ---------------------------------------------------------------------------
ARG L4T_TAG=r36.4.0-py3
ARG PYTHON_VERSION=3.11
ARG CUDA_ARCH_LIST="8.7"
ARG APP_USER=armdroid
ARG APP_UID=1000

# =============================================================================
# Stage 1: Builder — compile native wheels and install Python deps
# =============================================================================
FROM nvcr.io/nvidia/l4t-pytorch:${L4T_TAG} AS builder

ARG PYTHON_VERSION

# System build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        pkg-config \
        libusb-1.0-0-dev \
        librealsense2-dev \
        python${PYTHON_VERSION}-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only dependency metadata first (layer caching)
COPY pyproject.toml ./
COPY src/armdroid/__init__.py src/armdroid/
COPY src/armdroid/api/ src/armdroid/api/

# Install Python dependencies into a prefix we can copy later
RUN pip install --no-cache-dir --prefix=/install \
    ".[hardware,realsense]"

# Copy full source for the final install
COPY src/ src/
COPY config/ config/
COPY scripts/ scripts/
COPY assets/ assets/
COPY weights/ weights/
COPY firmware/ firmware/
COPY Makefile README.md LICENSE AGENTS.md ./

# Install armdroid itself into the prefix
RUN pip install --no-cache-dir --prefix=/install --no-deps .

# =============================================================================
# Stage 2: Runtime — minimal image with only runtime dependencies
# =============================================================================
FROM nvcr.io/nvidia/l4t-pytorch:${L4T_TAG} AS runtime

ARG APP_USER
ARG APP_UID

LABEL maintainer="ianshank <armdroid@github.com>"
LABEL description="Armdroid robot arm platform — Jetson deployment image"
LABEL version="0.2.0"

# Runtime-only system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        libusb-1.0-0 \
        librealsense2 \
        librealsense2-utils \
        udev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd --gid ${APP_UID} ${APP_USER} \
    && useradd --uid ${APP_UID} --gid ${APP_UID} --create-home ${APP_USER}

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application source and assets
WORKDIR /app
COPY --chown=${APP_USER}:${APP_USER} src/ src/
COPY --chown=${APP_USER}:${APP_USER} config/ config/
COPY --chown=${APP_USER}:${APP_USER} scripts/ scripts/
COPY --chown=${APP_USER}:${APP_USER} assets/ assets/
COPY --chown=${APP_USER}:${APP_USER} weights/ weights/
COPY --chown=${APP_USER}:${APP_USER} firmware/ firmware/
COPY --chown=${APP_USER}:${APP_USER} pyproject.toml Makefile README.md LICENSE ./

# Ensure the armdroid package is importable
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Serial device access: add user to dialout group
RUN usermod -aG dialout ${APP_USER}

# Health check — verifies GPU, armdroid import, and serial port
HEALTHCHECK --interval=30s --timeout=30s --start-period=45s --retries=3 \
    CMD ["python", "/app/scripts/jetson_health_check.py", "--probe-only"]

# Switch to non-root user
USER ${APP_USER}

# Default entrypoint: armdroid CLI
ENTRYPOINT ["python", "-m", "armdroid"]
CMD ["--help"]
