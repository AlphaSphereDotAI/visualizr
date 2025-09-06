# Deployment Guide

Complete guide for deploying Visualizr in various environments from development to
production scale.

## Table of Contents

- [Overview](#overview)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Production Deployment](#production-deployment)
- [Scaling & Load Balancing](#scaling--load-balancing)
- [Monitoring & Maintenance](#monitoring--maintenance)
- [Security Considerations](#security-considerations)

## Overview

Visualizr can be deployed in multiple environments:

1. **Local Development**: Direct installation for development
2. **Docker Containers**: Isolated deployment with Docker
3. **Cloud Platforms**: AWS, GCP, Azure deployment
4. **Production Servers**: High-availability production setup
5. **Kubernetes**: Scalable container orchestration

### Deployment Requirements

| Environment | CPU       | RAM   | GPU        | Storage | Network   |
|-------------|-----------|-------|------------|---------|-----------|
| Development | 4 cores   | 8GB   | Optional   | 10GB    | Local     |
| Testing     | 4 cores   | 8GB   | Optional   | 20GB    | Local/VPN |
| Staging     | 8 cores   | 16GB  | NVIDIA GPU | 50GB    | Private   |
| Production  | 16+ cores | 32GB+ | NVIDIA GPU | 100GB+  | Public    |

## Local Development

### Quick Setup

```bash
# Install with uv (recommended)
uvx --python 3.10 visualizr

# Or with pip in virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install visualizr

# Run locally
visualizr
```

### Development Configuration

```bash
# Create development environment file
cat > .env.development << EOF
VISUALIZR_HOST=127.0.0.1
VISUALIZR_PORT=7860
VISUALIZR_DEBUG=true
VISUALIZR_LOG_LEVEL=DEBUG
VISUALIZR_DEVICE=cuda
VISUALIZR_STEP_T=25
VISUALIZR_FACE_SR=false
VISUALIZR_RESULTS_DIR=./dev_results
VISUALIZR_ASSETS_DIR=./dev_assets
EOF

# Load and run
source .env.development
visualizr
```

### IDE Integration

#### VS Code Setup

.vscode/launch.json

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Visualizr Debug",
      "type": "python",
      "request": "launch",
      "module": "visualizr",
      "env": {
        "VISUALIZR_DEBUG": "true",
        "VISUALIZR_LOG_LEVEL": "DEBUG"
      },
      "console": "integratedTerminal",
      "justMyCode": false
    }
  ]
}
```

#### PyCharm Setup

```xml
<!-- Run Configuration -->
<component name="RunManager">
    <configuration name="Visualizr" type="PythonConfigurationType">
        <module name="visualizr"/>
        <option name="INTERPRETER_OPTIONS" value=""/>
        <option name="PARENT_ENVS" value="true"/>
        <envs>
            <env name="VISUALIZR_DEBUG" value="true"/>
            <env name="VISUALIZR_LOG_LEVEL" value="DEBUG"/>
        </envs>
    </configuration>
</component>
```

## Docker Deployment

### Basic Docker Setup

#### Using Pre-built Image

```bash
# Pull and run official image
docker pull visualizr:latest
docker run -p 7860:7860 --gpus all visualizr:latest

# With custom configuration
docker run -p 7860:7860 --gpus all \
    -e VISUALIZR_DEVICE=cuda \
    -e VISUALIZR_FACE_SR=true \
    -v $(pwd)/results:/app/results \
    -v $(pwd)/assets:/app/assets \
    visualizr:latest
```

#### Docker Compose (Recommended)

```yaml
# compose.yaml
version: '3.8'

services:
  visualizr:
    image: visualizr:latest
    ports:
      - "7860:7860"
    environment:
      - VISUALIZR_HOST=0.0.0.0
      - VISUALIZR_PORT=7860
      - VISUALIZR_DEVICE=cuda
      - VISUALIZR_LOG_LEVEL=INFO
    volumes:
      - ./results:/app/results
      - ./assets:/app/assets
      - ./models:/app/ckpts
      - ./logs:/app/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [ gpu ]
    restart: unless-stopped
    healthcheck:
      test: [ "CMD", "curl", "-f", "http://localhost:7860/heartbeat" ]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

volumes:
  models:
  results:
  assets:
  logs:
```

#### Custom Docker Build

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    curl \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1000 visualizr
USER visualizr
WORKDIR /app

# Python environment
RUN python3.10 -m venv venv
ENV PATH="/app/venv/bin:$PATH"

# Install uv for faster dependency resolution
RUN pip install uv

# Install Visualizr
RUN uv pip install visualizr

# Create directories
RUN mkdir -p {results,assets,ckpts,logs}

# Environment variables
ENV VISUALIZR_HOST=0.0.0.0
ENV VISUALIZR_PORT=7860
ENV VISUALIZR_DEVICE=cuda
ENV VISUALIZR_RESULTS_DIR=/app/results
ENV VISUALIZR_ASSETS_DIR=/app/assets
ENV VISUALIZR_CKPTS_DIR=/app/ckpts
ENV VISUALIZR_LOGS_DIR=/app/logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/heartbeat || exit 1

EXPOSE 7860

# Run application
CMD ["python", "-m", "visualizr"]
```

### Build and Deploy

```bash
# Build custom image
docker build -t visualizr:custom .

# Deploy with compose
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs -f visualizr
```

### Multi-stage Build (Production)

```dockerfile
# Multi-stage Dockerfile for production
FROM nvidia/cuda:11.8-devel-ubuntu22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3-pip git curl \
    && rm -rf /var/lib/apt/lists/*

# Build environment
WORKDIR /build
RUN python3.10 -m venv venv
ENV PATH="/build/venv/bin:$PATH"

# Install dependencies
RUN pip install uv
RUN uv pip install visualizr

# Production stage
FROM nvidia/cuda:11.8-runtime-ubuntu22.04

# Runtime dependencies only
RUN apt-get update && apt-get install -y \
    python3.10 curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy built environment
COPY --from=builder /build/venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# App user and directories
RUN useradd -m -u 1000 visualizr
USER visualizr
WORKDIR /app
RUN mkdir -p {results,assets,ckpts,logs}

# Production configuration
ENV VISUALIZR_HOST=0.0.0.0
ENV VISUALIZR_PORT=7860
ENV VISUALIZR_DEBUG=false
ENV VISUALIZR_LOG_LEVEL=INFO

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/heartbeat || exit 1

CMD ["python", "-m", "visualizr"]
```

## Cloud Deployment

### Amazon Web Services (AWS)

#### EC2 Instance Setup

```bash
#!/bin/bash
# AWS EC2 User Data Script

# Install Docker
yum update -y
yum install -y docker
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Install NVIDIA Docker (for GPU instances)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Deploy Visualizr
docker run -d \
    --name visualizr \
    --gpus all \
    -p 7860:7860 \
    -v /opt/visualizr/results:/app/results \
    -v /opt/visualizr/assets:/app/assets \
    -e VISUALIZR_HOST=0.0.0.0 \
    -e VISUALIZR_LOG_LEVEL=INFO \
    --restart unless-stopped \
    visualizr:latest
```

#### AWS ECS (Elastic Container Service)

```json
{
  "family": "visualizr",
  "networkMode": "awsvpc",
  "requiresCompatibilities": [
    "EC2"
  ],
  "cpu": "4096",
  "memory": "16384",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "visualizr",
      "image": "visualizr:latest",
      "portMappings": [
        {
          "containerPort": 7860,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "VISUALIZR_HOST",
          "value": "0.0.0.0"
        },
        {
          "name": "VISUALIZR_DEVICE",
          "value": "cuda"
        },
        {
          "name": "VISUALIZR_LOG_LEVEL",
          "value": "INFO"
        }
      ],
      "mountPoints": [
        {
          "sourceVolume": "results",
          "containerPath": "/app/results"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/visualizr",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "resourceRequirements": [
        {
          "type": "GPU",
          "value": "1"
        }
      ]
    }
  ],
  "volumes": [
    {
      "name": "results",
      "host": {
        "sourcePath": "/opt/visualizr/results"
      }
    }
  ]
}
```

#### AWS Lambda (CPU-only)

```python
# lambda_function.py
import json
import base64
import tempfile
from pathlib import Path
from visualizr.app.runner import app


def lambda_handler(event, context):
    """AWS Lambda handler for Visualizr"""

    try:
        # Parse request
        body = json.loads(event['body'])

        # Decode base64 files
        image_data = base64.b64decode(body['image'])
        audio_data = base64.b64decode(body['audio'])

        # Save to temporary files
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            image_path = tmp_path / "image.jpg"
            audio_path = tmp_path / "audio.wav"

            with open(image_path, 'wb') as f:
                f.write(image_data)
            with open(audio_path, 'wb') as f:
                f.write(audio_data)

            # Generate video
            result = app.generate_video(
                infer_type='mfcc_full_control',  # Fast for serverless
                image_path=str(image_path),
                audio_path=str(audio_path),
                face_sr=False,
                step_t=25,  # Reduced steps for speed
                # ... other parameters
            )

            video_256, video_512, message = result

            if video_256:
                # Read generated video
                with open(video_256.value, 'rb') as f:
                    video_data = f.read()

                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'success': True,
                        'video': base64.b64encode(video_data).decode(),
                        'message': message.value
                    })
                }
            else:
                return {
                    'statusCode': 500,
                    'body': json.dumps({
                        'success': False,
                        'error': message.value
                    })
                }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'success': False,
                'error': str(e)
            })
        }
```

### Google Cloud Platform (GCP)

#### Cloud Run Deployment

```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: [ 'build', '-t', 'gcr.io/$PROJECT_ID/visualizr', '.' ]
  - name: 'gcr.io/cloud-builders/docker'
    args: [ 'push', 'gcr.io/$PROJECT_ID/visualizr' ]
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'visualizr'
      - '--image'
      - 'gcr.io/$PROJECT_ID/visualizr'
      - '--platform'
      - 'managed'
      - '--region'
      - 'us-central1'
      - '--allow-unauthenticated'
      - '--memory'
      - '8Gi'
      - '--cpu'
      - '4'
      - '--timeout'
      - '600'
```

#### GKE (Google Kubernetes Engine)

```yaml
# gke-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: visualizr
spec:
  replicas: 2
  selector:
    matchLabels:
      app: visualizr
  template:
    metadata:
      labels:
        app: visualizr
    spec:
      containers:
        - name: visualizr
          image: alphaspheredotai/visualizr:latest
          ports:
            - containerPort: 7860
          env:
            - name: VISUALIZR_HOST
              value: "0.0.0.0"
            - name: VISUALIZR_DEVICE
              value: "cuda"
          resources:
            limits:
              nvidia.com/gpu: 1
              memory: "16Gi"
              cpu: "4"
            requests:
              memory: "8Gi"
              cpu: "2"
          volumeMounts:
            - name: results
              mountPath: /app/results
      volumes:
        - name: results
          persistentVolumeClaim:
            claimName: visualizr-results
      nodeSelector:
        accelerator: nvidia-tesla-t4
---
apiVersion: v1
kind: Service
metadata:
  name: visualizr-service
spec:
  selector:
    app: visualizr
  ports:
    - port: 80
      targetPort: 7860
  type: LoadBalancer
```

### Microsoft Azure

#### Azure Container Instances

```bash
# Deploy to Azure Container Instances
az container create \
    --resource-group visualizr-rg \
    --name visualizr-container \
    --image visualizr:latest \
    --cpu 4 \
    --memory 16 \
    --gpu-count 1 \
    --gpu-sku V100 \
    --ports 7860 \
    --environment-variables \
        VISUALIZR_HOST=0.0.0.0 \
        VISUALIZR_DEVICE=cuda \
    --azure-file-volume-account-name storageaccount \
    --azure-file-volume-account-key key \
    --azure-file-volume-share-name visualizr-share \
    --azure-file-volume-mount-path /app/results
```

#### Azure Kubernetes Service (AKS)

```yaml
# aks-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: visualizr
spec:
  replicas: 3
  selector:
    matchLabels:
      app: visualizr
  template:
    metadata:
      labels:
        app: visualizr
    spec:
      containers:
        - name: visualizr
          image: visualizr.azurecr.io/visualizr:latest
          ports:
            - containerPort: 7860
          resources:
            limits:
              nvidia.com/gpu: 1
              memory: "16Gi"
              cpu: "4000m"
            requests:
              memory: "8Gi"
              cpu: "2000m"
          env:
            - name: VISUALIZR_HOST
              value: "0.0.0.0"
            - name: VISUALIZR_DEVICE
              value: "cuda"
      nodeSelector:
        accelerator: nvidia
      tolerations:
        - key: sku
          operator: Equal
          value: gpu
          effect: NoSchedule
```

## Production Deployment

### Production-Ready Configuration

```yaml
# docker-compose.prod.yaml
version: '3.8'

services:
  visualizr:
    image: visualizr:latest
    restart: always
    ports:
      - "7860:7860"
    environment:
      - VISUALIZR_HOST=0.0.0.0
      - VISUALIZR_PORT=7860
      - VISUALIZR_DEBUG=false
      - VISUALIZR_LOG_LEVEL=INFO
      - VISUALIZR_DEVICE=cuda
      - VISUALIZR_MEMORY_FRACTION=0.8
      - VISUALIZR_ENABLE_MONITORING=true
    volumes:
      - results:/app/results
      - assets:/app/assets
      - models:/app/ckpts
      - logs:/app/logs
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 32G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [ gpu ]
    healthcheck:
      test: [ "CMD", "curl", "-f", "http://localhost:7860/heartbeat" ]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s
    logging:
      driver: json-file
      options:
        max-size: "100m"
        max-file: "5"

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - visualizr
    restart: always

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: always

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    restart: always

volumes:
  results:
  assets:
  models:
  logs:
  prometheus_data:
  grafana_data:
```

### Reverse Proxy (Nginx)

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream visualizr {
        server visualizr:7860;
        keepalive 32;
    }

    server {
        listen 80;
        server_name visualizr.example.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name visualizr.example.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

        # Rate limiting
        limit_req_zone $binary_remote_addr zone=api:10m rate=10r/m;
        limit_req_zone $binary_remote_addr zone=upload:10m rate=5r/m;

        # File upload limits
        client_max_body_size 50M;
        client_body_timeout 300s;

        location / {
            proxy_pass http://visualizr;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_cache_bypass $http_upgrade;

            # Timeouts for long-running requests
            proxy_connect_timeout 60s;
            proxy_send_timeout 300s;
            proxy_read_timeout 300s;
        }

        location /api/generate_video {
            limit_req zone=api burst=5 nodelay;
            proxy_pass http://visualizr;
            # Extended timeout for generation
            proxy_read_timeout 600s;
        }

        location /upload {
            limit_req zone=upload burst=2 nodelay;
            proxy_pass http://visualizr;
        }

        # Health check endpoint
        location /health {
            access_log off;
            proxy_pass http://visualizr/heartbeat;
        }

        # Static file serving
        location /static/ {
            alias /app/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
}
```

### SSL/TLS Configuration

```bash
# Generate SSL certificates with Let's Encrypt
docker run --rm -v $(pwd)/ssl:/etc/letsencrypt certbot/certbot \
    certonly \
    --standalone \
    -d visualizr.example.com \
    --email admin@example.com \
    --agree-tos \
    --non-interactive

# Or use existing certificates
cp /path/to/cert.pem ssl/
cp /path/to/key.pem ssl/
chmod 600 ssl/key.pem
```

## Scaling & Load Balancing

### Horizontal Scaling

#### Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Create overlay network
docker network create --driver overlay visualizr-network

# Deploy stack
docker stack deploy -c docker-compose.swarm.yaml visualizr-stack
```

```yaml
# docker-compose.swarm.yaml
version: '3.8'

services:
  visualizr:
    image: visualizr:latest
    deploy:
      replicas: 3
      placement:
        constraints:
          - node.role == worker
          - node.labels.gpu == true
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [ gpu ]
      update_config:
        parallelism: 1
        delay: 10s
        order: start-first
    networks:
      - visualizr-network
    environment:
      - VISUALIZR_HOST=0.0.0.0
      - VISUALIZR_DEVICE=cuda

  load_balancer:
    image: haproxy:alpine
    ports:
      - "80:80"
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.role == manager
    configs:
      - source: haproxy_config
        target: /usr/local/etc/haproxy/haproxy.cfg
    networks:
      - visualizr-network

networks:
  visualizr-network:
    driver: overlay
    attachable: true

configs:
  haproxy_config:
    external: true
```

#### Kubernetes Scaling

```yaml
# k8s-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: visualizr-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: visualizr
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Percent
          value: 50
          periodSeconds: 60
```

### Load Balancer Configuration

#### HAProxy

```bash
# haproxy.cfg
global
    daemon
    maxconn 4096
    log stdout local0

defaults
    mode http
    timeout connect 5s
    timeout client 300s
    timeout server 600s
    option httplog
    balance roundrobin

frontend visualizr_frontend
    bind *:80
    default_backend visualizr_backend

backend visualizr_backend
    option httpchk GET /heartbeat
    server viz1 visualizr_1:7860 check
    server viz2 visualizr_2:7860 check
    server viz3 visualizr_3:7860 check

listen stats
    bind *:8404
    stats enable
    stats uri /stats
    stats refresh 30s
```

#### Application Load Balancer (AWS)

```bash
# Create target group
aws elbv2 create-target-group \
    --name visualizr-targets \
    --protocol HTTP \
    --port 7860 \
    --vpc-id vpc-12345 \
    --health-check-path /heartbeat \
    --health-check-interval-seconds 30

# Create load balancer
aws elbv2 create-load-balancer \
    --name visualizr-lb \
    --subnets subnet-12345 subnet-67890 \
    --security-groups sg-12345
```

## Monitoring & Maintenance

### Health Monitoring

#### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'visualizr'
    static_configs:
      - targets: [ 'visualizr:7860' ]
    scrape_interval: 30s
    metrics_path: /metrics

  - job_name: 'node'
    static_configs:
      - targets: [ 'node-exporter:9100' ]

  - job_name: 'nvidia-gpu'
    static_configs:
      - targets: [ 'nvidia-gpu-exporter:9445' ]

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093
```

#### Custom Metrics Endpoint

```python
# Add to visualizr app for monitoring
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Metrics
generation_requests = Counter('visualizr_generations_total', 'Total generations')
generation_duration = Histogram('visualizr_generation_seconds', 'Generation time')
active_generations = Gauge('visualizr_active_generations', 'Active generations')
gpu_memory_usage = Gauge('visualizr_gpu_memory_bytes', 'GPU memory usage')


@generation_duration.time()
def generate_with_metrics(*args, **kwargs):
    generation_requests.inc()
    active_generations.inc()
    try:
        result = original_generate_video(*args, **kwargs)
        return result
    finally:
        active_generations.dec()
        # Update GPU memory usage
        if torch.cuda.is_available():
            gpu_memory_usage.set(torch.cuda.memory_allocated())


# Start metrics server
start_http_server(8000)
```

### Logging Strategy

#### Centralized Logging

```yaml
# docker-compose with logging
services:
  visualizr:
    logging:
      driver: fluentd
      options:
        fluentd-address: localhost:24224
        tag: visualizr.app

  fluentd:
    image: fluent/fluentd:v1.16-debian-1
    ports:
      - "24224:24224"
    volumes:
      - ./fluentd/conf:/fluentd/etc
      - ./logs:/fluentd/log
    environment:
      FLUENTD_CONF: fluent.conf

  elasticsearch:
    image: elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"

  kibana:
    image: kibana:8.11.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
```

#### Log Analysis Queries

```bash
# Common log analysis commands

# Error rate monitoring
grep "ERROR" /app/logs/visualizr.log | tail -100

# Generation performance
grep "Generation completed" /app/logs/visualizr.log | \
    grep -o "took [0-9.]*s" | \
    awk '{sum+=$2; count++} END {print "Avg:", sum/count "s"}'

# Memory usage patterns
grep "GPU memory" /app/logs/visualizr.log | \
    grep -o "[0-9.]*GB" | \
    sort -n | tail -10

# User activity patterns
grep "generate_video" /app/logs/visualizr.log | \
    cut -d' ' -f1 | uniq -c | sort -nr
```

### Backup & Recovery

#### Data Backup Strategy

```bash
#!/bin/bash
# backup.sh - Automated backup script

BACKUP_DIR="/backups/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Backup generated videos
rsync -av --progress /app/results/ "$BACKUP_DIR/results/"

# Backup user assets
rsync -av --progress /app/assets/ "$BACKUP_DIR/assets/"

# Backup configuration
cp /app/.env "$BACKUP_DIR/config.env"
cp /app/docker-compose.yaml "$BACKUP_DIR/"

# Backup logs (compressed)
tar -czf "$BACKUP_DIR/logs.tar.gz" /app/logs/

# Upload to cloud storage (AWS S3 example)
aws s3 sync "$BACKUP_DIR" "s3://visualizr-backups/$(date +%Y%m%d)/"

# Clean up old local backups (keep 7 days)
find /backups -type d -mtime +7 -exec rm -rf {} +

echo "Backup completed: $BACKUP_DIR"
```

#### Disaster Recovery

```bash
#!/bin/bash
# restore.sh - Disaster recovery script

RESTORE_DATE="${1:-$(date +%Y%m%d)}"
BACKUP_DIR="/backups/$RESTORE_DATE"

echo "Restoring from backup: $RESTORE_DATE"

# Stop services
docker-compose down

# Restore data
rsync -av "$BACKUP_DIR/results/" /app/results/
rsync -av "$BACKUP_DIR/assets/" /app/assets/

# Restore configuration
cp "$BACKUP_DIR/config.env" /app/.env
cp "$BACKUP_DIR/docker-compose.yaml" /app/

# Restore logs
tar -xzf "$BACKUP_DIR/logs.tar.gz" -C /

# Restart services
docker-compose up -d

echo "Restore completed"
```

## Security Considerations

### Access Control

#### Authentication Setup

```python
# Add authentication middleware
from functools import wraps
from flask import request, jsonify
import jwt


def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'No token provided'}), 401

        try:
            # Verify JWT token
            payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
            request.user = payload
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401

        return f(*args, **kwargs)

    return decorated


# Apply to generation endpoints
@require_auth
def generate_video_endpoint():
    # Protected generation logic
    pass
```

#### API Rate Limiting

```python
# Rate limiting with Redis
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import redis

redis_client = redis.Redis(host='redis', port=6379, db=0)

limiter = Limiter(
    app,
    key_func=get_remote_address,
    storage_uri="redis://redis:6379",
    default_limits=["100 per hour"]
)


@limiter.limit("5 per minute")
@app.route("/api/generate_video", methods=["POST"])
def rate_limited_generation():
    # Rate-limited generation
    pass
```

### Network Security

#### Firewall Rules

```bash
# UFW firewall configuration
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow from 10.0.0.0/8 to any port 7860  # Internal only
ufw enable
```

#### Docker Security

```yaml
# Security-hardened docker-compose
services:
  visualizr:
    image: visualizr:latest
    user: "1000:1000"  # Non-root user
    read_only: true
    tmpfs:
      - /tmp
      - /app/tmp
    volumes:
      - results:/app/results:rw
      - assets:/app/assets:ro
    cap_drop:
      - ALL
    cap_add:
      - CHOWN
      - SETGID
      - SETUID
    security_opt:
      - no-new-privileges:true
    sysctls:
      - net.core.somaxconn=1024
```

### Data Protection

#### Input Validation

```python
# Input validation and sanitization
from PIL import Image
import magic
from pathlib import Path


def validate_image(image_path: Path) -> bool:
    """Validate uploaded image file"""

    # Check file size (max 10MB)
    if image_path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("Image file too large")

    # Check file type using magic numbers
    file_type = magic.from_file(str(image_path), mime=True)
    if file_type not in ['image/jpeg', 'image/png']:
        raise ValueError("Invalid image format")

    # Validate image can be opened
    try:
        with Image.open(image_path) as img:
            img.verify()
    except Exception:
        raise ValueError("Corrupted image file")

    return True


def validate_audio(audio_path: Path) -> bool:
    """Validate uploaded audio file"""

    # Check file size (max 50MB)
    if audio_path.stat().st_size > 50 * 1024 * 1024:
        raise ValueError("Audio file too large")

    # Check file type
    file_type = magic.from_file(str(audio_path), mime=True)
    allowed_types = ['audio/wav', 'audio/mpeg', 'audio/mp4']
    if file_type not in allowed_types:
        raise ValueError("Invalid audio format")

    return True
```

#### Content Filtering

```python
# Content safety checks
import cv2
import numpy as np


def check_content_safety(image_path: Path) -> bool:
    """Basic content safety checks"""

    # Load image
    image = cv2.imread(str(image_path))

    # Check for minimum face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    faces = face_cascade.detectMultiScale(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        raise ValueError("No face detected in image")

    if len(faces) > 1:
        raise ValueError("Multiple faces detected")

    # Additional safety checks can be added here
    # - NSFW content detection
    # - Age verification
    # - Celebrity detection for privacy

    return True
```

---

*For troubleshooting deployment issues, see
the [Troubleshooting Guide](troubleshooting.md).*
