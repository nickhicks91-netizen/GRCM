Serving Module
==============

.. automodule:: grcm.bentoml_service
   :members:
   :undoc-members:
   :show-inheritance:

GRCMService
-----------

.. autoclass:: grcm.bentoml_service.GRCMService
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   BentoML service for production REST API deployment.

   **API Endpoints**:

   1. ``POST /predict`` - Single inference
   2. ``POST /predict_batch`` - Batch inference
   3. ``GET /health`` - Health check
   4. ``GET /metrics`` - Prometheus metrics

   **Usage**:

   Start the service:

   .. code-block:: bash

      python -m grcm.bentoml_service

      # Or with Docker
      docker-compose up grcm-api

   **API Examples**:

   **Single Prediction**:

   .. code-block:: bash

      curl -X POST http://localhost:3000/predict \
        -H "Content-Type: application/json" \
        -d '{
          "image_emb": [0.1, 0.2, ..., 0.5],  # 512-dim CLIP
          "audio_emb": [0.3, 0.1, ..., 0.8],  # 768-dim Wav2Vec2
          "action": [0.0, 1.0, 0.0, 0.0],     # Optional action
          "desire_idx": 0                      # 0=calm, 1=alert, 2=creative, 3=focus
        }'

   **Response**:

   .. code-block:: json

      {
        "coherence": 0.847,
        "phi": 2.134,
        "qualia": [0.45, 0.32, 0.18, 0.05],
        "desire_align": 0.812,
        "prop_state": [0.12, 0.34, ..., 0.56],
        "halt": false,
        "message": "Success"
      }

   **Batch Prediction**:

   .. code-block:: bash

      curl -X POST http://localhost:3000/predict_batch \
        -H "Content-Type: application/json" \
        -d '{
          "image_emb_batch": [[...], [...], [...]],  # [batch_size, 512]
          "audio_emb_batch": [[...], [...], [...]],  # [batch_size, 768]
          "desire_idx": 0
        }'

   **Response**:

   .. code-block:: json

      {
        "results": [
          {"coherence": 0.847, "phi": 2.134, ...},
          {"coherence": 0.792, "phi": 1.987, ...},
          {"coherence": 0.901, "phi": 2.456, ...}
        ],
        "aggregates": {
          "mean_coherence": 0.847,
          "mean_phi": 2.192,
          "coherence_above_threshold": 1.0
        }
      }

   **Health Check**:

   .. code-block:: bash

      curl http://localhost:3000/health

   **Response**:

   .. code-block:: json

      {
        "status": "healthy",
        "model_loaded": true,
        "device": "cpu"
      }

   **Prometheus Metrics**:

   .. code-block:: bash

      curl http://localhost:3000/metrics

   Returns Prometheus-formatted metrics including:

   - ``grcm_requests_total`` - Total requests
   - ``grcm_phi`` - Current phi value
   - ``grcm_coherence_mean`` - Mean coherence
   - ``grcm_qualia_*`` - Qualia state distributions
   - ``grcm_ethical_halt_count`` - Number of ethical halts

   **Methods**:

   .. automethod:: predict
   .. automethod:: predict_batch
   .. automethod:: health
   .. automethod:: metrics
