Deployment Guide
================

Deploy GRCM to production with Docker, Kubernetes, or cloud platforms.

Quick Start (Docker Compose)
-----------------------------

.. code-block:: bash

   # Clone repository
   git clone https://github.com/nickhicks91-netizen/GRCM.git
   cd GRCM/grcm-resonant

   # Start all services
   docker-compose up -d

   # Verify services
   docker-compose ps

   # Access services:
   # - GRCM API: http://localhost:3000
   # - MLflow UI: http://localhost:5000
   # - Gradio UI: http://localhost:7860
   # - Prometheus: http://localhost:9091
   # - Grafana: http://localhost:3001 (admin/admin)

Docker Deployment
-----------------

**Build Image**:

.. code-block:: bash

   docker build -t grcm-resonant:v0.1.0 .

**Run Container**:

.. code-block:: bash

   docker run -d \
     --name grcm-api \
     -p 3000:3000 \
     -p 9090:9090 \
     -v $(pwd)/configs:/app/configs:ro \
     -e GRCM_CONFIG=/app/configs/production.yaml \
     grcm-resonant:v0.1.0

**Test API**:

.. code-block:: bash

   curl -X POST http://localhost:3000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "image_emb": [...],
       "audio_emb": [...],
       "desire_idx": 0
     }'

Kubernetes Deployment
---------------------

**Prerequisites**:

- Kubernetes cluster (EKS, GKE, AKS, or local minikube)
- kubectl configured
- Docker registry access (ghcr.io)

**Deploy to Kubernetes**:

.. code-block:: bash

   # Create namespace
   kubectl create namespace grcm-prod

   # Apply deployment
   kubectl apply -f deployment/k8s/deployment.yaml
   kubectl apply -f deployment/k8s/ingress.yaml

   # Check status
   kubectl get pods -n grcm-prod
   kubectl get svc -n grcm-prod

   # Get external IP
   kubectl get svc grcm-api-service -n grcm-prod

**Scale Deployment**:

.. code-block:: bash

   # Manual scaling
   kubectl scale deployment grcm-api --replicas=5 -n grcm-prod

   # Horizontal Pod Autoscaler (HPA) is configured automatically
   kubectl get hpa -n grcm-prod

**Update Deployment**:

.. code-block:: bash

   # Build new version
   docker build -t ghcr.io/nickhicks91-netizen/grcm:v0.2.0 .
   docker push ghcr.io/nickhicks91-netizen/grcm:v0.2.0

   # Rolling update
   kubectl set image deployment/grcm-api \
     grcm-api=ghcr.io/nickhicks91-netizen/grcm:v0.2.0 \
     -n grcm-prod

   # Monitor rollout
   kubectl rollout status deployment/grcm-api -n grcm-prod

   # Rollback if needed
   kubectl rollout undo deployment/grcm-api -n grcm-prod

Cloud Platform Deployment
--------------------------

AWS ECS
~~~~~~~

.. code-block:: bash

   # Create ECR repository
   aws ecr create-repository --repository-name grcm-resonant

   # Build and push
   aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
   docker tag grcm-resonant:latest <account>.dkr.ecr.<region>.amazonaws.com/grcm-resonant:latest
   docker push <account>.dkr.ecr.<region>.amazonaws.com/grcm-resonant:latest

   # Create ECS task definition (see deployment/aws/task-definition.json)
   aws ecs register-task-definition --cli-input-json file://deployment/aws/task-definition.json

   # Create ECS service
   aws ecs create-service \
     --cluster grcm-cluster \
     --service-name grcm-api \
     --task-definition grcm-task \
     --desired-count 3

Google Cloud Run
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Build and push to GCR
   gcloud builds submit --tag gcr.io/<project-id>/grcm-resonant

   # Deploy to Cloud Run
   gcloud run deploy grcm-api \
     --image gcr.io/<project-id>/grcm-resonant \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 4Gi \
     --cpu 2

   # Get URL
   gcloud run services describe grcm-api --region us-central1

Azure Container Instances
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Login to Azure
   az login

   # Create resource group
   az group create --name grcm-rg --location eastus

   # Create container instance
   az container create \
     --resource-group grcm-rg \
     --name grcm-api \
     --image ghcr.io/nickhicks91-netizen/grcm:latest \
     --cpu 2 --memory 4 \
     --ports 3000 9090 \
     --ip-address Public

   # Get IP
   az container show --resource-group grcm-rg --name grcm-api --query ipAddress.ip

Monitoring Setup
----------------

**Prometheus** (metrics collection):

.. code-block:: bash

   # Add Prometheus Helm repo
   helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
   helm repo update

   # Install Prometheus
   helm install prometheus prometheus-community/prometheus \
     --namespace monitoring --create-namespace \
     --values deployment/prometheus-values.yaml

**Grafana** (visualization):

.. code-block:: bash

   # Install Grafana
   helm install grafana grafana/grafana \
     --namespace monitoring \
     --set adminPassword=admin

   # Import GRCM dashboards
   kubectl apply -f deployment/grafana/dashboards/

**Alerts**:

.. code-block:: bash

   # Apply Prometheus alert rules
   kubectl apply -f deployment/prometheus-alerts.yml

Production Checklist
--------------------

Before deploying to production:

.. code-block:: text

   ☐ Security
     ☐ Use non-root Docker user
     ☐ Enable TLS/HTTPS (Ingress with cert-manager)
     ☐ Set up API authentication
     ☐ Configure network policies
     ☐ Run security scans (Trivy, Snyk)

   ☐ Performance
     ☐ Enable torch.compile or TensorRT
     ☐ Configure HPA (CPU + custom metrics)
     ☐ Set resource requests/limits
     ☐ Enable FP16 on GPU
     ☐ Test latency under load

   ☐ Reliability
     ☐ Configure liveness/readiness probes
     ☐ Set up logging (ELK, CloudWatch, Stackdriver)
     ☐ Configure backups (model checkpoints, MLflow data)
     ☐ Test disaster recovery
     ☐ Document runbooks

   ☐ Monitoring
     ☐ Prometheus scraping enabled
     ☐ Grafana dashboards imported
     ☐ Alerts configured (PagerDuty, Slack)
     ☐ MLflow tracking configured
     ☐ Error tracking (Sentry)

Performance Tuning
------------------

**Latency Optimization**:

.. code-block:: yaml

   # config/production.yaml
   optimization:
     compile_mode: "max-autotune"  # Best performance
     use_fp16: true                # GPU only

**Throughput Optimization**:

.. code-block:: yaml

   serving:
     workers: 8           # Match CPU cores
     max_batch_size: 32   # Batch requests

**Resource Limits** (Kubernetes):

.. code-block:: yaml

   resources:
     requests:
       cpu: "2000m"       # 2 CPUs
       memory: "4Gi"
     limits:
       cpu: "4000m"       # 4 CPUs max
       memory: "8Gi"

Troubleshooting
---------------

**Pod CrashLoopBackOff**:

.. code-block:: bash

   kubectl logs <pod-name> -n grcm-prod
   kubectl describe pod <pod-name> -n grcm-prod

**High Latency**:

.. code-block:: bash

   # Check metrics
   kubectl top pods -n grcm-prod

   # Scale up
   kubectl scale deployment grcm-api --replicas=8 -n grcm-prod

**Low Coherence**:

.. code-block:: bash

   # Check Grafana dashboard
   # Adjust bandwidth in ConfigMap
   kubectl edit configmap grcm-config -n grcm-prod

For detailed troubleshooting, see :doc:`../deployment/runbooks/RUNBOOK`.

Next Steps
----------

- :doc:`../deployment/monitoring` - Set up monitoring
- :doc:`../deployment/runbooks` - Operations runbooks
- :doc:`../api/serving` - API reference
