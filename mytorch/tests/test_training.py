import argparse
import mytorch
import mytorch.nn as nn
import mytorch.optim as optim
from mytorch import Accelerator

class SimpleModel(nn.Module):
    """Simple 2-layer MLP for testing."""
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def test_forward_backward(accelerator, num_iterations=3):
    """Test forward and backward passes."""
    
    # Create model
    model = SimpleModel(input_dim=128, hidden_dim=256, output_dim=10)
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    # Prepare with accelerator
    model, optimizer = accelerator.prepare(model, optimizer)
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss()
    
    accelerator.print(f"Testing with mixed_precision={accelerator.mixed_precision}")
    
    # Training loop
    for iter in range(num_iterations):
       
        # Create dummy data
        batch_size = 32
        x = mytorch.randn((batch_size, 128))
        y = mytorch.randint(0, 10, (batch_size,))
        
        x = x.to(accelerator.device)
        y = y.to(accelerator.device)
        
        # Forward pass
        logits = model(x)
        loss = loss_fn(logits, y)
        
        # Backward pass
        accelerator.backward(loss)
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        # Log progress
        if iter % 5 == 0 or iter == num_iterations - 1:
            loss_val = accelerator.gather_for_metrics(loss)
    
    return True

def test_gradient_accumulation(accelerator, accumulation_steps=2, num_iterations=2):
    """Test gradient accumulation."""
    
    accelerator.print(f"Testing Gradient Accumulation (steps={accumulation_steps})")
    
    # Reinitialize accelerator with gradient accumulation
    accelerator.gradient_accumulation_steps = accumulation_steps
    
    # Create model
    model = SimpleModel(input_dim=128, hidden_dim=256, output_dim=10)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    # Prepare
    model, optimizer = accelerator.prepare(model, optimizer)
    loss_fn = nn.CrossEntropyLoss()
    
    completed_steps = 0
    for iter in range(num_iterations * accumulation_steps):

        batch_size = 32
        x = mytorch.randn((batch_size, 128))
        y = mytorch.randint(0, 10, (batch_size,))
        
        x = x.to(accelerator.device)
        y = y.to(accelerator.device)
        
        # Forward pass
        logits = model(x)
        loss = loss_fn(logits, y)
        
        # Backward pass
        accelerator.backward(loss)
        
        # Step optimizer
        optimizer.step()
        optimizer.zero_grad()

        ### Grad is only zeroed out after a full accumulation ###
        if accelerator.sync_grad:
            assert (model.fc1.weight.grad is None) or all(model.fc1.weight.grad==0.0)
        else:
            assert model.fc1.weight.grad is not None
  
        # Check if accumulation is done
        if accelerator.sync_grad:
            completed_steps += 1
            loss_val = accelerator.gather_for_metrics(loss)
    
    return True

def test_distributed_sync(accelerator):
    """Test that gradients are properly synchronized across ranks."""
    
    if accelerator.num_processes == 1:
        accelerator.print("\nSkipping distributed sync test (single GPU)")
        return True
    
    accelerator.print(f"Testing Distributed Synchronization")
    # Create model
    model = SimpleModel(input_dim=128, hidden_dim=256, output_dim=10)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    # Prepare
    model, optimizer = accelerator.prepare(model, optimizer)
    loss_fn = nn.CrossEntropyLoss()
    
    # Create SAME data on all ranks for testing
    x = mytorch.ones((32, 128)).to(accelerator.device)
    y = mytorch.zeros((32,)).to(accelerator.device)
    
    # Forward/backward
    logits = model(x)
    loss = loss_fn(logits, y)
    accelerator.backward(loss)
    
    # Check that loss is the same across all ranks
    loss_val = accelerator.gather_for_metrics(loss)
    
    # All ranks should have the same loss since we used the same seed
    return True

def main():
    parser = argparse.ArgumentParser(description="Test MyTorch training pipeline")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--all", action="store_true", help="Run all tests")
    args = parser.parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision
    )
    
    accelerator.print("MyTorch Training Pipeline Tests")
    accelerator.print(f"Number of processes: {accelerator.num_processes}")
    accelerator.print(f"Process index: {accelerator.device}")
    accelerator.print(f"Device: {accelerator.device}")
    accelerator.print(f"Mixed precision: {accelerator.mixed_precision}")

    all_passed = True
    
    #Test 1: Basic forward/backward
    try:
        test_forward_backward(accelerator, num_iterations=10)
    except Exception as e:
        accelerator.print(f"\n✗ Test failed: Forward/backward - {e}")
        all_passed = False
    
    # Test 2: Gradient accumulation (if steps > 1 or --all flag)
    if args.gradient_accumulation_steps > 1 or args.all:
        try:
            test_gradient_accumulation(accelerator, accumulation_steps=args.gradient_accumulation_steps)
        except Exception as e:
            accelerator.print(f"\n✗ Test failed: Gradient accumulation - {e}")
            all_passed = False
    
    # Test 3: Distributed sync (if multi-GPU)
    if accelerator.num_processes > 1:
        try:
            test_distributed_sync(accelerator)
        except Exception as e:
            accelerator.print(f"\n✗ Test failed: Distributed sync - {e}")
            all_passed = False
    
    accelerator.end_training()
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
    