# HeapKit: Browser Heap Exploitation Toolkit

HeapKit is a Python-based toolkit that models Chrome's PartitionAlloc2 allocator, 
providing utilities for:

- Analyzing allocator behavior
- Generating JavaScript heap sprays
- Visualizing freelist states
- Simulating common heap bugs (UAF, overflow, double free)
- Providing exploitation strategy recommendations

## Features

- **PartitionAlloc Modeling**: Simplified, high-level Python simulation of Chrome’s allocator
- **JavaScript Sprays**: Code generation for ArrayBuffers, Typed Arrays, Objects, Strings, etc.
- **Freelist Analysis**: Track and visualize how free slots are reused
- **Bug Simulation**: Basic modeling for UAF, overflow, and double-free scenarios
- **Strategy Generation**: Automated heap grooming sequences, recommended exploitation objects
