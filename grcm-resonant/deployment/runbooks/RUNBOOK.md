# GRCM Operations Runbook

Standard operating procedures for GRCM production deployment.

## Table of Contents

1. [Deployment Procedures](#deployment-procedures)
2. [Incident Response](#incident-response)
3. [Maintenance](#maintenance)
4. [Disaster Recovery](#disaster-recovery)

---

## Deployment Procedures

### New Version Deployment

**Checklist**:
- [ ] Code reviewed and approved
- [ ] Tests passing (CI/CD green)
- [ ] Changelog updated
- [ ] Benchmarks completed
- [ ] Backup current state

**Steps**:

1. **Build and Tag Image**
```bash
# Build new version
docker build -t grcm-resonant:v0.2.0 .

# Tag for registry
docker tag grcm-resonant:v0.2.0 ghcr.io/nickhicks91-netizen/grcm:v0.2.0

# Push to registry
docker push ghcr.io/nickhicks91-netizen/grcm:v0.2.0
```

2. **Deploy to Staging**
```bash
# Update image
kubectl set image deployment/grcm-api grcm-api=grcm-resonant:v0.2.0 -n grcm-staging

# Monitor rollout
kubectl rollout status deployment/grcm-api -n grcm-staging

# Run smoke tests
kubectl run smoke-test --image=curlimages/curl --rm -it --restart=Never -- \
  curl -f http://grcm-api-service/health
```

3. **Deploy to Production**
```bash
# Blue-green deployment (zero downtime)
kubectl set image deployment/grcm-api grcm-api=grcm-resonant:v0.2.0 -n grcm-prod

# Monitor metrics in Grafana
# Watch for:
# - Error rate spike
# - Latency increase
# - Coherence degradation
# - Memory divergence

# If issues detected:
kubectl rollout undo deployment/grcm-api -n grcm-prod
```

### Rollback Procedure

```bash
# 1. Immediate rollback
kubectl rollout undo deployment/grcm-api -n grcm-prod

# 2. Rollback to specific version
kubectl rollout history deployment/grcm-api -n grcm-prod
kubectl rollout undo deployment/grcm-api --to-revision=5 -n grcm-prod

# 3. Verify rollback
kubectl rollout status deployment/grcm-api -n grcm-prod
kubectl get pods -n grcm-prod

# 4. Check metrics
# Coherence should stabilize
# Error rate should decrease
# Phi should return to normal range
```

---

## Incident Response

### High Error Rate Alert

**Symptoms**:
- Prometheus alert: `HighErrorRate`
- Error rate >10% for 5+ minutes

**Triage**:
```bash
# 1. Check pod logs
kubectl logs -l app=grcm --tail=100 -n grcm-prod

# 2. Check recent deployments
kubectl rollout history deployment/grcm-api -n grcm-prod

# 3. Check resource usage
kubectl top pods -n grcm-prod

# 4. Check external dependencies
curl http://mlflow:5000/health
```

**Common Causes**:
1. **Recent deployment issue**: Rollback
2. **Resource exhaustion**: Scale up
3. **Dependency failure**: Check MLflow, storage
4. **Bad input data**: Review API requests

**Resolution**:
```bash
# If deployment issue
kubectl rollout undo deployment/grcm-api -n grcm-prod

# If resource issue
kubectl scale deployment grcm-api --replicas=10 -n grcm-prod

# If dependency issue
kubectl restart deployment mlflow -n grcm-prod
```

### Low Coherence Alert

**Symptoms**:
- Prometheus alert: `LowCoherenceRate`
- >70% of samples with coherence <0.7

**Diagnosis**:
```bash
# 1. Check Grafana dashboard
# Look at coherence trend over time

# 2. Sample recent inputs
kubectl logs -l app=grcm --tail=50 -n grcm-prod | grep "coherence"

# 3. Check desire state distribution
# Ensure diverse desire states being used
```

**Possible Causes**:
1. **Poor input quality**: Random or corrupted embeddings
2. **Model drift**: Needs retraining (EchoMirror)
3. **Wrong desire state**: Mismatch with input modality
4. **Configuration issue**: Bandwidth too narrow

**Resolution**:
```bash
# 1. Validate input sources
# Check CLIP/Wav2Vec embedding quality

# 2. Adjust bandwidth (if appropriate)
kubectl edit configmap grcm-config -n grcm-prod
# Increase base_bandwidth from 0.5 to 0.6

# 3. Retrain with EchoMirror
python examples/mlflow_demo.py
# Use real EEG/voice data

# 4. Monitor recovery
# Coherence should improve within 10-15 minutes
```

### Ethical Halt Spike

**Symptoms**:
- Prometheus alert: `HighDissonanceRate`
- >50% requests triggering ethical halts

**This is a FEATURE, not a bug!**

**Analysis**:
```bash
# 1. Check qualia distribution
kubectl logs -l app=grcm --tail=100 -n grcm-prod | grep "qualia"

# 2. Review input patterns
# What changed in input data?

# 3. Check recent model updates
# Was EchoMirror recently run?
```

**Actions**:
1. **Investigation**: Determine why inputs are conflicting
2. **User notification**: Alert users of uncertainty
3. **Data review**: Examine inputs triggering halts
4. **Model tuning**: May need EchoMirror adjustment

**NOT recommended**: Disabling halt mechanism (removes ethical safeguard)

### Memory Divergence

**Symptoms**:
- Prometheus alert: `MemoryDivergence`
- Memory norm >100 for 15+ minutes

**Critical Issue - Immediate Action Required**

```bash
# 1. Check memory state
kubectl logs -l app=grcm --tail=50 -n grcm-prod | grep "memory_norm"

# 2. Immediate mitigation: Restart pod
kubectl delete pod -l app=grcm -n grcm-prod --grace-period=30

# 3. Check for memory leaks
kubectl top pod -n grcm-prod --containers

# 4. Review recent inputs
# Look for NaN or Inf values
```

**Root Cause Investigation**:
1. Input validation failure (NaN/Inf inputs)
2. Numerical instability in forward pass
3. Memory leak in PyTorch graph
4. GRU cell divergence

**Prevention**:
- Add input sanitization
- Lower learning rate in EchoMirror
- Monitor phi std (should be <1.0)

### Pod Crash Loop

**Symptoms**:
- Pods continuously restarting
- `CrashLoopBackOff` status

**Triage**:
```bash
# 1. Check pod events
kubectl describe pod <pod-name> -n grcm-prod

# 2. Check logs (current and previous)
kubectl logs <pod-name> -n grcm-prod
kubectl logs <pod-name> -n grcm-prod --previous

# 3. Check resource limits
kubectl top pod <pod-name> -n grcm-prod
```

**Common Causes**:
1. **OOM (Out of Memory)**: Increase memory limit
2. **Startup timeout**: Increase `initialDelaySeconds` in liveness probe
3. **Missing dependencies**: Check image build
4. **Configuration error**: Validate ConfigMap

**Resolution**:
```bash
# If OOM
kubectl edit deployment grcm-api -n grcm-prod
# Increase memory limit to 4Gi

# If startup timeout
# Edit liveness probe initialDelaySeconds to 60s

# If config error
kubectl logs <pod-name> -n grcm-prod
# Fix ConfigMap and redeploy
```

---

## Maintenance

### Routine Maintenance

**Weekly**:
- [ ] Review Grafana dashboards
- [ ] Check error logs for patterns
- [ ] Review resource usage trends
- [ ] Update dependencies (security patches)

**Monthly**:
- [ ] Review and tune autoscaling policies
- [ ] Analyze phi stability trends
- [ ] Run EchoMirror retraining (if needed)
- [ ] Review and update alerts
- [ ] Backup model checkpoints

**Quarterly**:
- [ ] Performance benchmarking
- [ ] Disaster recovery drill
- [ ] Review and update runbooks
- [ ] Capacity planning

### Model Retraining (EchoMirror)

```bash
# 1. Collect training data
# EEG theta/alpha + voice spectra + labels

# 2. Run training
python examples/mlflow_demo.py

# 3. Evaluate in MLflow UI
mlflow ui
# Compare phi, coherence, desire alignment

# 4. Export trained model
python -c "
from grcm import ResonantConsciousnessModule
import torch
model = ResonantConsciousnessModule(15, 8, 32)
# Load trained desires
torch.save(model.state_dict(), 'grcm_tuned.pth')
"

# 5. Deploy new model
# Update ConfigMap or mount as volume
kubectl create configmap grcm-model --from-file=grcm_tuned.pth -n grcm-prod
```

### Scaling Operations

**Scale Up** (increased load):
```bash
# Manual scale
kubectl scale deployment grcm-api --replicas=8 -n grcm-prod

# Or update HPA
kubectl edit hpa grcm-api-hpa -n grcm-prod
# Increase maxReplicas
```

**Scale Down** (cost optimization):
```bash
# During low-traffic periods
kubectl scale deployment grcm-api --replicas=2 -n grcm-prod

# Update HPA for off-hours
# Use CronJob to adjust replicas
```

---

## Disaster Recovery

### Backup Strategy

**What to Backup**:
1. Model weights (desire vectors, trained parameters)
2. MLflow experiment data
3. Configuration files
4. Prometheus data (metrics history)

**Backup Procedure**:
```bash
# 1. Backup PVC (models)
kubectl get pvc grcm-models-pvc -n grcm-prod -o yaml > grcm-pvc-backup.yaml

# 2. Backup ConfigMaps
kubectl get configmap grcm-config -n grcm-prod -o yaml > grcm-config-backup.yaml

# 3. Backup MLflow data
kubectl exec -it mlflow-0 -n grcm-prod -- tar -czf /tmp/mlflow-backup.tar.gz /mlflow
kubectl cp grcm-prod/mlflow-0:/tmp/mlflow-backup.tar.gz ./mlflow-backup.tar.gz

# 4. Store backups in S3/GCS/Azure Blob
aws s3 cp mlflow-backup.tar.gz s3://grcm-backups/$(date +%Y%m%d)/
```

### Recovery Procedure

**Complete Cluster Failure**:

1. **Restore Infrastructure**
```bash
# Recreate cluster (see DEPLOYMENT_GUIDE.md)

# Restore namespace
kubectl create namespace grcm-prod
```

2. **Restore Configuration**
```bash
# Apply ConfigMaps
kubectl apply -f grcm-config-backup.yaml

# Restore PVC
kubectl apply -f grcm-pvc-backup.yaml
```

3. **Restore Data**
```bash
# Restore MLflow data
kubectl run restore-pod --image=busybox -n grcm-prod -- sleep 3600
kubectl cp mlflow-backup.tar.gz grcm-prod/restore-pod:/tmp/
kubectl exec -it restore-pod -n grcm-prod -- tar -xzf /tmp/mlflow-backup.tar.gz -C /mlflow
```

4. **Redeploy Applications**
```bash
kubectl apply -f deployment/k8s/deployment.yaml
kubectl rollout status deployment/grcm-api -n grcm-prod
```

5. **Verify Recovery**
```bash
# Health check
curl http://<external-ip>/health

# Run test predictions
curl -X POST http://<external-ip>/predict \
  -H "Content-Type: application/json" \
  -d @test_payload.json

# Check metrics in Grafana
```

**Recovery Time Objective (RTO)**: <30 minutes
**Recovery Point Objective (RPO)**: <1 hour

---

## Escalation Matrix

| Severity | Response Time | Escalate To |
|----------|--------------|-------------|
| **P0** (Production Down) | Immediate | On-call engineer + Manager |
| **P1** (Degraded Service) | <15 min | On-call engineer |
| **P2** (Minor Issue) | <1 hour | Team slack channel |
| **P3** (Monitoring Alert) | <4 hours | Create ticket |

**On-Call Contacts**:
- Primary: [On-call rotation]
- Secondary: [Team lead]
- Manager: [Engineering manager]

---

## Useful Commands

```bash
# Quick pod status
kubectl get pods -n grcm-prod -o wide

# Watch deployments
kubectl get deployments -n grcm-prod --watch

# Tail all pod logs
kubectl logs -f -l app=grcm -n grcm-prod --all-containers=true

# Port-forward for local testing
kubectl port-forward svc/grcm-api-service 3000:80 -n grcm-prod

# Execute shell in pod
kubectl exec -it <pod-name> -n grcm-prod -- /bin/bash

# Check resource quotas
kubectl describe resourcequota -n grcm-prod

# List all resources
kubectl get all -n grcm-prod
```

---

**Last Updated**: 2025-11-14
**Maintainer**: GRCM Operations Team
