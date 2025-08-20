def test_subpixel1d():
    """Test SubPixel1D layer"""
    print("Testing SubPixel1D...")

    # Test data: (batch=2, width=8, channels=4) -> should become (batch=2, width=16, channels=2)
    x = torch.randn(2, 8, 4)
    r = 2

    try:
        output = SubPixel1D(x, r)
        expected_shape = (2, 16, 2)

        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print(f"✓ SubPixel1D: {x.shape} -> {output.shape}")

        # Test layer wrapper
        layer = SubPixel1DLayer(upscale_factor=2)
        output2 = layer(x)
        assert torch.equal(output, output2), "Layer wrapper should give same result"
        print("✓ SubPixel1DLayer wrapper works")

    except Exception as e:
        print(f"✗ SubPixel1D failed: {e}")
        return False
    return True

def test_transformer_components():
    """Test Transformer components"""
    print("\nTesting Transformer components...")

    try:
        batch_size, seq_len, embed_dim = 2, 16, 128
        x = torch.randn(batch_size, seq_len, embed_dim)

        # Test MultiHeadSelfAttention
        attention = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=8)
        attn_output = attention(x)
        assert attn_output.shape == x.shape, f"Attention output shape mismatch: {attn_output.shape} vs {x.shape}"
        print(f"✓ MultiHeadSelfAttention: {x.shape} -> {attn_output.shape}")

        # Test EncoderLayer
        encoder = EncoderLayer(embed_dim=embed_dim, num_heads=8, ff_dim=512)
        enc_output = encoder(x, training=True)
        assert enc_output.shape == x.shape, f"Encoder output shape mismatch: {enc_output.shape} vs {x.shape}"
        print(f"✓ EncoderLayer: {x.shape} -> {enc_output.shape}")

        # Test TransformerBlock
        transformer = TransformerBlock(num_layers=2, embed_dim=embed_dim,
                                     maximum_position_encoding=32, num_heads=8)
        trans_output = transformer(x, training=True)
        assert trans_output.shape == x.shape, f"Transformer output shape mismatch: {trans_output.shape} vs {x.shape}"
        print(f"✓ TransformerBlock: {x.shape} -> {trans_output.shape}")

    except Exception as e:
        print(f"✗ Transformer components failed: {e}")
        return False
    return True

def test_afilm_layer():
    """Test AFiLM layer"""
    print("\nTesting AFiLM layer...")

    try:
        batch_size, n_step, n_filters = 2, 512, 128
        block_size = 32
        x = torch.randn(batch_size, n_step, n_filters)

        afilm = AFiLM(n_step=n_step, block_size=block_size, n_filters=n_filters)
        output = afilm(x)

        assert output.shape == x.shape, f"AFiLM output shape mismatch: {output.shape} vs {x.shape}"
        print(f"✓ AFiLM: {x.shape} -> {output.shape}")

    except Exception as e:
        print(f"✗ AFiLM layer failed: {e}")
        return False
    return True

def test_afilm_model():
    """Test complete AFiLM model"""
    print("\nTesting complete AFiLM model...")

    try:
        # Create model
        model = get_afilm(n_layers=4, scale=4)

        # Test input (same as original: batch=1, length=8192, channels=1)
        x = torch.randn(1, 8192, 1)

        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(x)

        # Should output same shape as input
        assert output.shape == x.shape, f"Model output shape mismatch: {output.shape} vs {x.shape}"
        print(f"✓ AFiLM Model: {x.shape} -> {output.shape}")

        # Test with batch size > 1
        x_batch = torch.randn(4, 8192, 1)
        with torch.no_grad():
            output_batch = model(x_batch)
        assert output_batch.shape == x_batch.shape, f"Batch output shape mismatch: {output_batch.shape} vs {x_batch.shape}"
        print(f"✓ AFiLM Model (batch): {x_batch.shape} -> {output_batch.shape}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Model parameters: {trainable_params:,} trainable / {total_params:,} total")

    except Exception as e:
        print(f"✗ AFiLM model failed: {e}")
        return False
    return True

def test_utils():
    """Test utility functions"""
    print("\nTesting utility functions...")

    try:
        # Test spline_up
        x_lr = np.random.randn(100)
        r = 4
        x_hr = spline_up(x_lr, r)

        expected_len = len(x_lr) * r
        assert len(x_hr) == expected_len, f"Spline upsampling length mismatch: {len(x_hr)} vs {expected_len}"
        print(f"✓ spline_up: {len(x_lr)} -> {len(x_hr)} (factor {r})")

    except Exception as e:
        print(f"✗ Utils failed: {e}")
        return False
    return True

def test_model_serialization():
    """Test model save/load"""
    print("\nTesting model serialization...")

    try:
        # Create and save model
        model = get_afilm(n_layers=2, scale=4)  # Smaller model for faster testing

        # Test data
        x = torch.randn(1, 8192, 1)
        model.eval()
        with torch.no_grad():
            output1 = model(x)

        # Save model
        torch.save(model.state_dict(), 'test_model.pth')

        # Create new model and load
        model2 = get_afilm(n_layers=2, scale=4)
        model2.load_state_dict(torch.load('test_model.pth', map_location='cpu'))
        model2.eval()

        # Test same output
        with torch.no_grad():
            output2 = model2(x)

        assert torch.allclose(output1, output2, atol=1e-6), "Loaded model gives different output"
        print("✓ Model serialization works")

        # Clean up
        os.remove('test_model.pth')

    except Exception as e:
        print(f"✗ Model serialization failed: {e}")
        return False
    return True

def run_all_tests():
    """Run all component tests"""
    print("="*50)
    print("RUNNING AFILM PYTORCH CONVERSION TESTS")
    print("="*50)

    tests = [
        test_subpixel1d,
        test_transformer_components,
        test_afilm_layer,
        test_afilm_model,
        test_utils,
        test_model_serialization
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("="*50)
    print(f"RESULTS: {passed}/{total} tests passed")
    if passed == total:
        print("ALL TESTS PASSED!")
    else:
        print("Some tests failed.")
    print("="*50)

    return passed == total


run_all_tests()