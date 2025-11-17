#!/usr/bin/env python
"""
Test script to verify FisherTune implementation

This script checks:
1. All modules import correctly
2. Configurations are created properly
3. FisherTune components initialize without errors
4. Basic functionality works
"""

import sys
import torch
import numpy as np


def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")

    try:
        from model.fishertune import (
            FisherTuneConfig,
            FisherTuneModule,
            DomainPerturbation,
            get_experiment_config
        )
        print("  ✓ fishertune module imported")
    except Exception as e:
        print(f"  ✗ fishertune import failed: {e}")
        return False

    try:
        from model.fishertune_efficient import (
            EfficientFisherComputer,
            AdaptiveScheduler,
            BlockWiseFisher,
            estimate_training_overhead,
            print_efficiency_report
        )
        print("  ✓ fishertune_efficient module imported")
    except Exception as e:
        print(f"  ✗ fishertune_efficient import failed: {e}")
        return False

    try:
        from model.fishertune_trainer import FisherTuneTrainer
        print("  ✓ fishertune_trainer module imported")
    except Exception as e:
        print(f"  ✗ fishertune_trainer import failed: {e}")
        return False

    try:
        from model.UniCDR import UniCDR, BehaviorAggregator
        print("  ✓ UniCDR module imported")
    except Exception as e:
        print(f"  ✗ UniCDR import failed: {e}")
        return False

    return True


def test_experiment_configs():
    """Test experiment configuration creation"""
    print("\nTesting experiment configurations...")

    from model.fishertune import get_experiment_config

    experiment_types = [
        'baseline', 'fim_only', 'dr_fim', 'unified',
        'shared_only', 'specific_only', 'adaptive',
        'perturbation_edge_dropout', 'perturbation_popularity',
        'perturbation_noise', 'perturbation_cross_domain'
    ]

    all_passed = True
    for exp_type in experiment_types:
        try:
            config = get_experiment_config(exp_type)
            print(f"  ✓ {exp_type}: mode={config.param_mode}, perturbation={config.perturbation_type}")
        except Exception as e:
            print(f"  ✗ {exp_type} failed: {e}")
            all_passed = False

    return all_passed


def test_unicdr_model():
    """Test UniCDR model creation"""
    print("\nTesting UniCDR model creation...")

    try:
        from model.UniCDR import UniCDR

        # Create minimal configuration
        opt = {
            "num_domains": 2,
            "user_max": [100, 100],
            "item_max": [200, 200],
            "latent_dim": 64,
            "dropout": 0.3,
            "aggregator": "mean",
            "lambda_a": 0.5,
            "task": "dual-user-intra",
            "cuda": False
        }

        model = UniCDR(opt)
        print(f"  ✓ UniCDR model created")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Total parameters: {total_params:,}")

        # Check parameter structure
        shared_count = 0
        specific_count = 0
        for name, param in model.named_parameters():
            if 'share' in name:
                shared_count += param.numel()
            else:
                specific_count += param.numel()

        print(f"  ✓ Shared params: {shared_count:,}")
        print(f"  ✓ Specific params: {specific_count:,}")

        return True

    except Exception as e:
        print(f"  ✗ UniCDR creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fishertune_module():
    """Test FisherTune module initialization"""
    print("\nTesting FisherTune module...")

    try:
        from model.UniCDR import UniCDR
        from model.fishertune import FisherTuneModule, FisherTuneConfig

        # Create model
        opt = {
            "num_domains": 2,
            "user_max": [100, 100],
            "item_max": [200, 200],
            "latent_dim": 64,
            "dropout": 0.3,
            "aggregator": "mean",
            "lambda_a": 0.5,
            "task": "dual-user-intra",
            "cuda": False
        }
        model = UniCDR(opt)

        # Create FisherTune config
        config = FisherTuneConfig(
            use_fim=True,
            use_dr_fim=True,
            param_mode='adaptive',
            device='cpu'
        )

        # Initialize FisherTune
        ft_module = FisherTuneModule(model, config)
        print(f"  ✓ FisherTuneModule initialized")
        print(f"  ✓ Shared params: {len(ft_module.shared_params)}")
        print(f"  ✓ Specific params: {len(ft_module.specific_params)}")

        # Test mask update
        ft_module.update_parameter_masks(10)
        print(f"  ✓ Parameter masks updated")

        # Test statistics
        stats = ft_module.get_statistics()
        print(f"  ✓ Statistics: {stats}")

        return True

    except Exception as e:
        print(f"  ✗ FisherTune module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_perturbation():
    """Test perturbation strategies"""
    print("\nTesting perturbation strategies...")

    try:
        from model.fishertune import DomainPerturbation, FisherTuneConfig

        opt = {
            "num_domains": 2
        }

        # Create dummy batch
        batch_size = 4
        maxlen = 10
        num_neg = 5

        user = torch.randint(0, 100, (batch_size,))
        pos_item = torch.randint(0, 200, (batch_size,))
        neg_item = torch.randint(0, 200, (batch_size, num_neg))
        context_item = torch.randint(0, 200, (batch_size, maxlen))
        context_score = torch.rand(batch_size, maxlen)
        global_item = torch.randint(0, 200, (batch_size, 2, maxlen))
        global_score = torch.rand(batch_size, 2, maxlen)

        batch = (user, pos_item, neg_item, context_item, context_score, global_item, global_score)

        # Test each perturbation type
        perturbation_types = ['none', 'edge_dropout', 'popularity_weight', 'noise', 'cross_domain']

        for p_type in perturbation_types:
            config = FisherTuneConfig(perturbation_type=p_type, device='cpu')
            perturbation = DomainPerturbation(config, opt)

            perturbed_batch = perturbation.perturb_batch(batch, domain_id=0)

            # Check batch structure preserved
            assert len(perturbed_batch) == 7
            assert perturbed_batch[0].shape == user.shape
            print(f"  ✓ {p_type} perturbation works")

        return True

    except Exception as e:
        print(f"  ✗ Perturbation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_efficient_fisher():
    """Test efficient Fisher computation"""
    print("\nTesting efficient Fisher computation...")

    try:
        from model.fishertune_efficient import (
            EfficientFisherComputer,
            AdaptiveScheduler,
            estimate_training_overhead
        )
        from model.fishertune import FisherTuneConfig
        from model.UniCDR import UniCDR

        # Create model
        opt = {
            "num_domains": 2,
            "user_max": [100, 100],
            "item_max": [200, 200],
            "latent_dim": 64,
            "dropout": 0.3,
            "aggregator": "mean",
            "lambda_a": 0.5,
            "task": "dual-user-intra",
            "cuda": False
        }
        model = UniCDR(opt)

        # Test efficient Fisher computer
        efficient_fisher = EfficientFisherComputer(model, device='cpu')
        print(f"  ✓ EfficientFisherComputer initialized")

        # Simulate gradient update
        criterion = torch.nn.BCEWithLogitsLoss()
        pred = torch.randn(10)
        target = torch.ones(10)
        loss = criterion(pred, target)
        loss.backward()

        # Update Fisher
        efficient_fisher.update_fisher_online(loss)
        print(f"  ✓ Online Fisher update works")

        # Test adaptive scheduler
        scheduler = AdaptiveScheduler()
        threshold = scheduler.compute_threshold(efficient_fisher.get_fisher())
        print(f"  ✓ AdaptiveScheduler threshold: {threshold:.6f}")

        scheduler.update_schedule(0.5)
        print(f"  ✓ Scheduler ratio: {scheduler.current_ratio:.4f}")

        # Test overhead estimation
        config = FisherTuneConfig()
        overhead = estimate_training_overhead(config, 1000000, 2)
        print(f"  ✓ Estimated overhead: {overhead:.2f}%")

        return True

    except Exception as e:
        print(f"  ✗ Efficient Fisher test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer_initialization():
    """Test FisherTune trainer initialization (without data)"""
    print("\nTesting FisherTune trainer initialization...")

    try:
        from model.fishertune_trainer import FisherTuneTrainer
        from model.fishertune import get_experiment_config

        opt = {
            "model": "UniCDR",
            "num_domains": 2,
            "user_max": [100, 100],
            "item_max": [200, 200],
            "latent_dim": 64,
            "dropout": 0.3,
            "aggregator": "mean",
            "lambda_a": 0.5,
            "lambda_loss": 0.4,
            "task": "dual-user-intra",
            "cuda": False,
            "optim": "adam",
            "lr": 0.001,
            "weight_decay": 1e-5,
            "device": torch.device('cpu')
        }

        # Test different modes
        modes = ['baseline', 'fim_only', 'dr_fim', 'adaptive']

        for mode in modes:
            ft_config = get_experiment_config(mode)
            ft_config.device = 'cpu'

            trainer = FisherTuneTrainer(opt, ft_config)
            print(f"  ✓ Trainer initialized with mode={mode}, FT enabled={trainer.ft_enabled}")

        return True

    except Exception as e:
        print(f"  ✗ Trainer initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("FISHERTUNE IMPLEMENTATION TESTS")
    print("=" * 60)

    results = {}

    results['imports'] = test_imports()
    results['configs'] = test_experiment_configs()
    results['unicdr'] = test_unicdr_model()
    results['fishertune'] = test_fishertune_module()
    results['perturbation'] = test_perturbation()
    results['efficient'] = test_efficient_fisher()
    results['trainer'] = test_trainer_initialization()

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n🎉 All tests passed! FisherTune implementation is ready.\n")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
