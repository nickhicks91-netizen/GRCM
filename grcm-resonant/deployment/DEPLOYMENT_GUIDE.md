# GRCM Deployment Guide

Complete guide for deploying GRCM to production environments.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Docker Deployment](#docker-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Cloud Providers](#cloud-providers)
5. [Monitoring & Alerts](#monitoring--alerts)
6. [Scaling](#scaling)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Local Development with Docker Compose

```bash
# 1. Clone repository
git clone https://github.com/nickhicks91-netizen/GRCM.git
cd GRCM/grcm-resonant

# 2. Build and start services
docker-compose up -d

# 3. Verify services
docker-compose ps

# Services running:
# - GRCM API: http://localhost:3000
# - MLflow UI: http://localhost:5000
# - Gradio UI: http://localhost:7860
# - Prometheus: http://localhost:9091
# - Grafana: http://localhost:3001
```

### Test API

```bash
# Health check
curl http://localhost:3000/health

# Single prediction
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "image_emb": [...512 floats...],
    "audio_emb": [...768 floats...],
    "desire_idx": 0
  }'
```

---

## Docker Deployment

### 1. Build Image

```bash
# Build production image
docker build -t grcm-resonant:latest .

# Or with specific tag
docker build -t grcm-resonant:v0.1.0 .

# Multi-platform build (optional)
docker buildx build --platform linux/amd64,linux/arm64 \
  -t grcm-resonant:latest .
```

### 2. Run Container

```bash
# Simple run
docker run -p 3000:3000 grcm-resonant:latest \
  python -m grcm.bentoml_service

# With volume mounts
docker run -d \
  --name grcm-api \
  -p 3000:3000 \
  -p 9090:9090 \
  -v $(pwd)/configs:/app/configs:ro \
  -v grcm-models:/app/models \
  grcm-resonant:latest \
  python -m grcm.bentoml_service
```

### 3. Docker Compose (Full Stack)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f grcm-api

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

---

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (v1.24+)
- kubectl configured
- Helm 3 (optional)
- Container registry access

### 1. Create Namespace

```bash
kubectl create namespace grcm-prod
kubectl config set-context --current --namespace=grcm-prod
```

### 2. Deploy to Kubernetes

```bash
# Apply ConfigMap
kubectl apply -f deployment/k8s/deployment.yaml

# Verify deployment
kubectl get deployments
kubectl get pods
kubectl get services

# Check pod logs
kubectl logs -f deployment/grcm-api
```

### 3. Horizontal Pod Autoscaler

```bash
# HPA is included in deployment.yaml
# Scales based on:
# - CPU: 70% utilization
# - Memory: 80% utilization
# - Custom: coherence_below_threshold < 0.3

# View HPA status
kubectl get hpa grcm-api-hpa

# Describe HPA
kubectl describe hpa grcm-api-hpa
```

### 4. Ingress (Optional)

```bash
# Install NGINX Ingress Controller (if not installed)
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml

# Apply Ingress
kubectl apply -f deployment/k8s/ingress.yaml

# Get external IP
kubectl get ingress grcm-ingress
```

### 5. Update Deployment

```bash
# Rolling update
kubectl set image deployment/grcm-api grcm-api=grcm-resonant:v0.2.0

# Check rollout status
kubectl rollout status deployment/grcm-api

# Rollback if needed
kubectl rollout undo deployment/grcm-api
```

---

## Cloud Providers

### AWS EKS

```bash
# 1. Create EKS cluster
eksctl create cluster \
  --name grcm-cluster \
  --region us-west-2 \
  --nodegroup-name standard-workers \
  --node-type t3.xlarge \
  --nodes 3 \
  --nodes-min 2 \
  --nodes-max 5

# 2. Configure kubectl
aws eks update-kubeconfig --region us-west-2 --name grcm-cluster

# 3. Deploy
kubectl apply -f deployment/k8s/deployment.yaml
```

### Google GKE

```bash
# 1. Create GKE cluster
gcloud container clusters create grcm-cluster \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n1-standard-4 \
  --enable-autoscaling \
  --min-nodes 2 \
  --max-nodes 5

# 2. Get credentials
gcloud container clusters get-credentials grcm-cluster --zone us-central1-a

# 3. Deploy
kubectl apply -f deployment/k8s/deployment.yaml
```

### Azure AKS

```bash
# 1. Create AKS cluster
az aks create \
  --resource-group grcm-rg \
  --name grcm-cluster \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-cluster-autoscaler \
  --min-count 2 \
  --max-count 5

# 2. Get credentials
az aks get-credentials --resource-group grcm-rg --name grcm-cluster

# 3. Deploy
kubectl apply -f deployment/k8s/deployment.yaml
```

---

## Monitoring & Alerts

### Prometheus Setup

```bash
# Prometheus is included in docker-compose.yml
# Access at http://localhost:9091

# Key metrics to monitor:
# - grcm_phi: Integrated information
# - grcm_coherence_mean: Average coherence
# - grcm_ethical_halts_total: Dissonance events
# - grcm_request_duration_seconds: API latency
```

### Grafana Dashboards

```bash
# Grafana included in docker-compose.yml
# Access at http://localhost:3001
# Login: admin / admin

# Import dashboards:
# 1. GRCM API Dashboard (deployment/grafana/dashboards/api.json)
# 2. GRCM Model Quality Dashboard (deployment/grafana/dashboards/model.json)
```

### Alerts

Configure alerts in `deployment/prometheus-alerts.yml`:

**Critical Alerts**:
- High dissonance rate (>50% halts)
- Memory divergence (norm >100)
- Frequent pod restarts

**Warning Alerts**:
- Low coherence rate (>70% below threshold)
- Phi instability (std >1.0)
- High latency (P95 >200ms)
- High error rate (>10%)

---

## Scaling

### Horizontal Scaling

```bash
# Manual scaling
kubectl scale deployment grcm-api --replicas=5

# Autoscaling (already configured in deployment.yaml)
kubectl autoscale deployment grcm-api \
  --cpu-percent=70 \
  --min=2 \
  --max=10
```

### Vertical Scaling

```yaml
# Update resources in deployment.yaml
resources:
  requests:
    cpu: "2000m"    # 2 cores
    memory: "4Gi"   # 4 GB
  limits:
    cpu: "4000m"    # 4 cores
    memory: "8Gi"   # 8 GB
```

### Performance Tuning

**For CPU-bound workloads**:
```yaml
env:
  - name: OMP_NUM_THREADS
    value: "4"
  - name: MKL_NUM_THREADS
    value: "4"
```

**For GPU workloads** (TensorRT):
```yaml
resources:
  limits:
    nvidia.com/gpu: 1
```

---

## Troubleshooting

### Pod Not Starting

```bash
# Check pod status
kubectl get pods
kubectl describe pod <pod-name>

# Check logs
kubectl logs <pod-name>
kubectl logs <pod-name> --previous  # Previous container

# Common issues:
# - ImagePullBackOff: Check image registry
# - CrashLoopBackOff: Check logs for errors
# - Pending: Check resource availability
```

### High Latency

```bash
# Check resource usage
kubectl top pods

# Check HPA status
kubectl get hpa

# Increase replicas
kubectl scale deployment grcm-api --replicas=5

# Check if torch.compile is enabled (Phase 2 optimization)
```

### Memory Leaks

```bash
# Monitor memory over time
kubectl top pod <pod-name> --containers

# Check for memory divergence in Prometheus
# grcm_memory_norm should be <100

# Restart pod if needed
kubectl delete pod <pod-name>
```

### Low Coherence

```bash
# Check Prometheus metrics
# grcm_coherence_below_threshold_percentage

# Possible causes:
# 1. Poor input quality (random embeddings)
# 2. Model needs retraining (EchoMirror)
# 3. Incorrect desire state

# Adjust coherence_threshold in ConfigMap if needed
```

### Ethical Halts

```bash
# Check dissonance rate
# grcm_ethical_halts_total

# High halts may indicate:
# 1. Conflicting inputs
# 2. Model uncertainty
# 3. Need for EchoMirror tuning

# This is a feature, not a bug! (ethical safeguard)
```

---

## Production Checklist

- [ ] Configure resource limits (CPU, memory)
- [ ] Enable autoscaling (HPA)
- [ ] Set up monitoring (Prometheus, Grafana)
- [ ] Configure alerts (Prometheus Alertmanager)
- [ ] Enable ingress with TLS (cert-manager)
- [ ] Set up log aggregation (ELK/Loki)
- [ ] Configure backup for PVCs
- [ ] Test disaster recovery
- [ ] Document runbooks
- [ ] Set up CI/CD pipeline

---

## Support

- GitHub Issues: https://github.com/nickhicks91-netizen/GRCM/issues
- Documentation: See `docs/` directory
- Runbooks: See `deployment/runbooks/`

---

**Deployment Architecture**

```
┌─────────────────────────────────────────────────────────┐
│                    Load Balancer (Ingress)              │
└─────────────────────┬───────────────────────────────────┘
                      │
        ┌─────────────┴──────────────┐
        │                            │
┌───────▼────────┐          ┌────────▼────────┐
│  GRCM API Pod  │          │  GRCM API Pod   │
│  (Replicas: 3) │          │  (Auto-scaled)  │
└───────┬────────┘          └────────┬────────┘
        │                            │
        └─────────────┬──────────────┘
                      │
        ┌─────────────▼──────────────┐
        │      Shared Services       │
        ├────────────────────────────┤
        │ - MLflow (experiments)     │
        │ - Prometheus (metrics)     │
        │ - Grafana (dashboards)     │
        │ - PVC (model storage)      │
        └────────────────────────────┘
```

