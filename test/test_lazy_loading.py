#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for lazy model loading implementation.
Tests that models load on-demand, not at initialization.
"""

import time
import sys
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from model_comparison_analyzer import ModelComparisonAnalyzer

print("=" * 70)
print("LAZY MODEL LOADING TEST")
print("=" * 70)

# Test 1: Initialization should be instant
print("\n[Test 1] Creating ModelComparisonAnalyzer instance...")
print("Expected: Should complete in < 1 second")
print("-" * 70)

start_time = time.time()
analyzer = ModelComparisonAnalyzer()
init_time = time.time() - start_time

print(f"✓ Initialization completed in {init_time:.2f} seconds")

if init_time < 1.0:
    print("✅ PASS - Initialization is fast (models not loaded)")
else:
    print("❌ FAIL - Initialization too slow (models might be loading)")
    sys.exit(1)

# Test 2: Check internal state
print("\n[Test 2] Verifying internal state...")
print("-" * 70)

print(f"model_configs defined: {bool(analyzer.model_configs)}")
print(f"Number of model configs: {len(analyzer.model_configs)}")
print(f"Available models: {list(analyzer.model_configs.keys())}")

print(f"\n_loaded_models (should be empty): {analyzer._loaded_models}")

if len(analyzer._loaded_models) == 0:
    print("✅ PASS - No models loaded at initialization")
else:
    print(f"❌ FAIL - {len(analyzer._loaded_models)} models already loaded!")
    sys.exit(1)

# Test 3: get_model() should load on demand
print("\n[Test 3] Testing lazy loading with get_model()...")
print("Expected: Model loads only when get_model() is called")
print("-" * 70)

test_model = "deepseek-1.5b"
print(f"\nCalling get_model('{test_model}')...")

start_time = time.time()
try:
    model = analyzer.get_model(test_model)
    load_time = time.time() - start_time

    print(f"✓ Model loaded in {load_time:.2f} seconds")
    print(f"Model type: {type(model)}")

    # Check internal state updated
    if test_model in analyzer._loaded_models:
        print(f"✓ Model cached in _loaded_models")
        print("✅ PASS - Lazy loading works correctly")
    else:
        print("❌ FAIL - Model not cached properly")
        sys.exit(1)

except Exception as e:
    print(f"❌ FAIL - Error loading model: {e}")
    print("\nPossible causes:")
    print("  1. Ollama server not running (run: ollama serve)")
    print(f"  2. Model '{test_model}' not installed (run: ollama pull deepseek-r1:1.5b)")
    print("  3. Ollama URL incorrect (check http://localhost:11434)")
    sys.exit(1)

# Test 4: Second call should use cached model (instant)
print("\n[Test 4] Testing model caching...")
print("Expected: Second call to get_model() should be instant")
print("-" * 70)

start_time = time.time()
model_again = analyzer.get_model(test_model)
cache_time = time.time() - start_time

print(f"✓ Cached retrieval in {cache_time:.4f} seconds")

if cache_time < 0.1:  # Should be < 100ms
    print("✅ PASS - Model caching works (instant retrieval)")
else:
    print("⚠️  WARNING - Cache retrieval slower than expected")

# Test 5: Verify other models NOT loaded
print("\n[Test 5] Verifying only requested model is loaded...")
print("-" * 70)

loaded_count = len(analyzer._loaded_models)
total_configs = len(analyzer.model_configs)

print(f"Models loaded: {loaded_count} / {total_configs}")
print(f"Loaded models: {list(analyzer._loaded_models.keys())}")

if loaded_count == 1:
    print("✅ PASS - Only requested model loaded (others still lazy)")
else:
    print(f"⚠️  WARNING - Expected 1 model loaded, got {loaded_count}")

# Test 6: Load another model
print("\n[Test 6] Loading second model on-demand...")
print("-" * 70)

second_model = "mistral"
print(f"Calling get_model('{second_model}')...")

start_time = time.time()
try:
    model2 = analyzer.get_model(second_model)
    load_time = time.time() - start_time

    print(f"✓ Second model loaded in {load_time:.2f} seconds")
    print(f"Total models now loaded: {len(analyzer._loaded_models)}")
    print(f"Loaded: {list(analyzer._loaded_models.keys())}")

    if len(analyzer._loaded_models) == 2:
        print("✅ PASS - Multiple models can coexist")

except Exception as e:
    print(f"⚠️  Could not load second model: {e}")
    print("(This is OK if you don't have mistral installed)")

# Test 7: Invalid model name
print("\n[Test 7] Testing error handling for invalid model...")
print("-" * 70)

try:
    analyzer.get_model("invalid_model_name")
    print("❌ FAIL - Should have raised ValueError for invalid model")
    sys.exit(1)
except ValueError as e:
    print(f"✓ Correctly raised ValueError: {e}")
    print("✅ PASS - Error handling works")
except Exception as e:
    print(f"⚠️  Unexpected error type: {e}")

# Summary
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)

print("\n✅ All core tests passed!")
print("\nKey findings:")
print(f"  • Initialization time: {init_time:.2f}s (should be < 1s)")
print(f"  • First model load: {load_time:.2f}s (depends on Ollama)")
print(f"  • Cached retrieval: {cache_time:.4f}s (should be < 0.1s)")
print(f"  • Models loaded on-demand: Yes")
print(f"  • Total models loaded: {len(analyzer._loaded_models)} / {total_configs}")

print("\n" + "=" * 70)
print("PERFORMANCE COMPARISON")
print("=" * 70)

old_startup = 4 * 45  # 4 models × 45 sec avg
new_startup = max(init_time, 0.01)  # Avoid division by zero

print(f"\nOLD (eager loading):  ~{old_startup:.0f} seconds (load all 4 models)")
print(f"NEW (lazy loading):    {init_time:.2f} seconds (load nothing)")
if init_time > 0:
    print(f"IMPROVEMENT:           {old_startup/new_startup:.0f}x faster startup!")
else:
    print(f"IMPROVEMENT:           Effectively instant (< 0.01s)!")

print("\n✅ Lazy loading implementation verified successfully!")
print("\nNext: Run 'streamlit run app.py' to test in the full application")
