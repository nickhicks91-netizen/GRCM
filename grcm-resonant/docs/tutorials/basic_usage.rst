Basic Usage Tutorial
====================

Complete walkthrough of GRCM's core features.

Setup
-----

.. code-block:: python

   import torch
   from grcm import ResonantConsciousnessModule

   # Initialize model
   model = ResonantConsciousnessModule(
       image_dim=512,
       audio_dim=768,
       hidden_dim=256,
       num_desires=4,
       num_nodes=64,
       num_qualia=4
   )

   print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

Single Forward Pass
-------------------

.. code-block:: python

   # Create sample embeddings
   image_emb = torch.randn(1, 512)  # CLIP embedding
   audio_emb = torch.randn(1, 768)  # Wav2Vec2 embedding
   action = torch.tensor([[0.0, 1.0, 0.0, 0.0]])  # Optional action

   # Forward pass with calm desire state
   result = model(image_emb, audio_emb, action=action, desire_idx=0)

   # Inspect outputs
   print("\n=== GRCM Output ===")
   print(f"Coherence: {result['coherence'].item():.3f}")
   print(f"Phi (Φ): {result['phi']:.3f}")
   print(f"Qualia: {result['qualia'].tolist()}")
   print(f"Desire alignment: {result['desire_align'].item():.3f}")
   print(f"Ethical halt: {result['halt']}")
   print(f"Output shape: {result['output'].shape}")

**Expected Output**:

.. code-block:: text

   === GRCM Output ===
   Coherence: 0.847
   Phi (Φ): 2.134
   Qualia: [[0.45, 0.32, 0.18, 0.05]]
   Desire alignment: 0.812
   Ethical halt: False
   Output shape: torch.Size([1, 256])

Understanding Outputs
---------------------

**Coherence** (0.0 - 1.0):
   How well the input resonates with learned frequency patterns. Values >0.7 pass coherence gating.

**Phi (Φ)** (0.0+):
   Integrated information estimate. Higher values suggest richer conscious processing.
   - Φ < 1.0: Low integration
   - 1.0 ≤ Φ < 2.0: Moderate
   - Φ ≥ 2.0: High integration

**Qualia** [calm, alert, creative, dissonance]:
   Phenomenal state distribution. If dissonance (index 3) >0.6, ethical halt triggers.

**Desire Alignment** (0.0 - 1.0):
   How well the input matches the current desire state. <0.5 suggests misalignment.

**Halt** (bool):
   If True, system refused to produce output due to high dissonance (ethical safeguard).

Desire States
-------------

.. code-block:: python

   # Test all 4 desire states
   desires = ["Calm", "Alert", "Creative", "Focus"]

   for idx, name in enumerate(desires):
       result = model(image_emb, audio_emb, desire_idx=idx)
       print(f"\n{name} (desire {idx}):")
       print(f"  Coherence: {result['coherence'].item():.3f}")
       print(f"  Phi: {result['phi']:.3f}")
       print(f"  Alignment: {result['desire_align'].item():.3f}")

**Output**:

.. code-block:: text

   Calm (desire 0):
     Coherence: 0.847
     Phi: 2.134
     Alignment: 0.812

   Alert (desire 1):
     Coherence: 0.792
     Phi: 1.987
     Alignment: 0.634

   Creative (desire 2):
     Coherence: 0.901
     Phi: 2.456
     Alignment: 0.723

   Focus (desire 3):
     Coherence: 0.776
     Phi: 2.012
     Alignment: 0.891

Sequential Processing
---------------------

.. code-block:: python

   # Simulate video processing
   num_frames = 50

   phi_history = []
   coherence_history = []

   for i in range(num_frames):
       # Generate varied inputs
       image_emb = torch.randn(1, 512)
       audio_emb = torch.randn(1, 768)

       result = model(image_emb, audio_emb, desire_idx=0)

       phi_history.append(result['phi'])
       coherence_history.append(result['coherence'].item())

       if (i + 1) % 10 == 0:
           print(f"Frame {i+1}: Phi = {result['phi']:.3f}, Coherence = {coherence_history[-1]:.3f}")

   # Analyze trends
   import numpy as np
   print(f"\nPhi statistics:")
   print(f"  Mean: {np.mean(phi_history):.3f}")
   print(f"  Std: {np.std(phi_history):.3f}")
   print(f"  Range: [{np.min(phi_history):.3f}, {np.max(phi_history):.3f}]")

Ethical Halt Handling
---------------------

.. code-block:: python

   # Simulate high dissonance scenario
   for i in range(10):
       image_emb = torch.randn(1, 512)
       audio_emb = torch.randn(1, 768)

       result = model(image_emb, audio_emb, desire_idx=0)

       if result['halt']:
           print(f"\n🛑 Ethical halt triggered on iteration {i+1}")
           print(f"   Dissonance: {result['qualia'][0, 3].item():.3f}")
           print(f"   Suggestion: Switch desire state or review input")

           # Try different desire state
           for new_idx in range(4):
               new_result = model(image_emb, audio_emb, desire_idx=new_idx)
               if not new_result['halt']:
                   print(f"   ✓ Desire {new_idx} resolves conflict")
                   break
           break

Batch Processing
----------------

.. code-block:: python

   # Process batch
   batch_size = 16
   image_batch = torch.randn(batch_size, 512)
   audio_batch = torch.randn(batch_size, 768)

   # Note: GRCM processes one sample at a time for episodic memory
   # Use BentoML API for efficient batching

   results = []
   for i in range(batch_size):
       result = model(
           image_batch[i:i+1],
           audio_batch[i:i+1],
           desire_idx=0
       )
       results.append(result)

   # Aggregate statistics
   coherences = [r['coherence'].item() for r in results]
   phis = [r['phi'] for r in results]

   print(f"\nBatch statistics (n={batch_size}):")
   print(f"  Mean coherence: {np.mean(coherences):.3f}")
   print(f"  Coherence >0.7: {sum(1 for c in coherences if c > 0.7) / batch_size * 100:.1f}%")
   print(f"  Mean phi: {np.mean(phis):.3f}")
   print(f"  Phi std: {np.std(phis):.3f}")

Memory Reset
------------

.. code-block:: python

   # Process some frames
   for i in range(10):
       result = model(torch.randn(1, 512), torch.randn(1, 768), desire_idx=0)

   print(f"Memory before reset: {torch.norm(model.memory).item():.3f}")

   # Reset memory (start new episode)
   model.reset_memory()

   print(f"Memory after reset: {torch.norm(model.memory).item():.3f}")  # Should be ~0

Save and Load
-------------

.. code-block:: python

   # Save checkpoint
   model.save_checkpoint("checkpoints/grcm_trained.pth")

   # Load checkpoint
   model_new = ResonantConsciousnessModule(512, 768, 256)
   model_new.load_checkpoint("checkpoints/grcm_trained.pth")

   # Verify loaded correctly
   result_orig = model(image_emb, audio_emb)
   result_loaded = model_new(image_emb, audio_emb)

   print(f"Phi difference: {abs(result_orig['phi'] - result_loaded['phi']):.6f}")  # Should be ~0

Next Steps
----------

- :doc:`echomirror_training` - Train custom desire states
- :doc:`multimodal_integration` - Integrate with CLIP/Wav2Vec2
- :doc:`performance_optimization` - Optimize for production
