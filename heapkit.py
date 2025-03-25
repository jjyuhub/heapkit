#!/usr/bin/env python3
"""
HeapKit: Browser Heap Exploitation Toolkit (Stand-Alone Single-File Version)

This script consolidates all the modules and functionality into a single file.
You can run it directly with:

    python heapkit.py [command] [options]

Use the --help flag to see available commands and options.
"""

import argparse
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
from partition_alloc_adv import PartitionRoot

###############################################################################
#                              ALLOCATOR MODULE
###############################################################################

class SlotSpan:
    """Represents a slot span in PartitionAlloc."""
    
    def __init__(self, size_class, num_slots):
        self.size_class = size_class
        self.num_slots = num_slots
        self.slots = [None] * num_slots  # None = free, object = allocated
        self.freelist = list(range(num_slots))  # Indices of free slots
    
    def allocate(self, object_type):
        """Allocate a slot and place object_type in it."""
        if not self.freelist:
            return None  # No free slots
        
        slot_idx = self.freelist.pop(0)  # Get first free slot
        self.slots[slot_idx] = object_type
        return slot_idx
    
    def free(self, slot_idx):
        """Free a slot."""
        if self.slots[slot_idx] is None:
            raise ValueError(f"Slot {slot_idx} is already free")
        
        self.slots[slot_idx] = None
        self.freelist.append(slot_idx)  # LIFO or FIFO depends on analysis; here it's appended
    
    def occupancy(self):
        """Return the occupancy percentage of this span."""
        allocated = sum(1 for slot in self.slots if slot is not None)
        return allocated / self.num_slots * 100


class Bucket:
    """Represents a bucket in PartitionAlloc."""
    
    def __init__(self, size_class):
        self.size_class = size_class
        self.slot_spans = []
        self.active_slot_span_idx = None
    
    def allocate(self, object_type):
        """Allocate an object in this bucket."""
        # If we have an active slot span, try to allocate there first
        if self.active_slot_span_idx is not None:
            slot_idx = self.slot_spans[self.active_slot_span_idx].allocate(object_type)
            if slot_idx is not None:
                return (self.active_slot_span_idx, slot_idx)
        
        # Try existing slot spans
        for i, span in enumerate(self.slot_spans):
            slot_idx = span.allocate(object_type)
            if slot_idx is not None:
                self.active_slot_span_idx = i
                return (i, slot_idx)
        
        # Need a new slot span
        new_span_idx = len(self.slot_spans)
        # Simplified approach: number of slots depends on size_class
        num_slots = max(1, 4096 // self.size_class)
        new_span = SlotSpan(self.size_class, num_slots)
        self.slot_spans.append(new_span)
        
        slot_idx = new_span.allocate(object_type)
        self.active_slot_span_idx = new_span_idx
        return (new_span_idx, slot_idx)
    
    def free(self, span_idx, slot_idx):
        """Free an allocated object."""
        self.slot_spans[span_idx].free(slot_idx)
        
        if self.active_slot_span_idx is None:
            self.active_slot_span_idx = span_idx


class PartitionAlloc:
    """Models Chrome's PartitionAlloc2 allocator behavior (simplified)."""
    
    DEFAULT_SIZE_CLASSES = [
        8, 16, 32, 48, 64, 80, 96, 112, 128, 
        144, 160, 176, 192, 208, 224, 240, 256,
        288, 320, 352, 384, 416, 448, 480, 512,
        576, 640, 704, 768, 832, 896, 960, 1024,
        1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048,
        2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096,
        4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192,
        9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384
    ]
    
    def __init__(self, size_classes=None):
        self.size_classes = size_classes or self.DEFAULT_SIZE_CLASSES
        self.buckets = {size: Bucket(size) for size in self.size_classes}
        self.allocations = {}  # Maps allocation ID -> (size_class, span_idx, slot_idx)
        self.next_id = 1
    
    def get_size_class(self, size):
        """Find the appropriate size class for requested size."""
        for size_class in self.size_classes:
            if size <= size_class:
                return size_class
        return self.size_classes[-1]
    
    def allocate(self, size, object_type):
        """Allocate an object of given size and type."""
        size_class = self.get_size_class(size)
        bucket = self.buckets[size_class]
        
        allocation = bucket.allocate(object_type)
        if allocation is None:
            return None
        
        alloc_id = self.next_id
        self.next_id += 1
        
        span_idx, slot_idx = allocation
        self.allocations[alloc_id] = (size_class, span_idx, slot_idx)
        
        return alloc_id
    
    def free(self, alloc_id):
        """Free an allocated object."""
        if alloc_id not in self.allocations:
            raise ValueError(f"Invalid allocation ID: {alloc_id}")
        
        size_class, span_idx, slot_idx = self.allocations[alloc_id]
        bucket = self.buckets[size_class]
        bucket.free(span_idx, slot_idx)
        del self.allocations[alloc_id]
    
    def get_allocation_info(self, alloc_id):
        """Get information about an allocation."""
        if alloc_id not in self.allocations:
            return None
        
        size_class, span_idx, slot_idx = self.allocations[alloc_id]
        bucket = self.buckets[size_class]
        span = bucket.slot_spans[span_idx]
        object_type = span.slots[slot_idx]
        
        return {
            'id': alloc_id,
            'size_class': size_class,
            'span_idx': span_idx,
            'slot_idx': slot_idx,
            'object_type': object_type
        }
    
    def get_bucket_stats(self, size_class):
        """Get statistics for a bucket."""
        if size_class not in self.buckets:
            return None
        
        bucket = self.buckets[size_class]
        spans = bucket.slot_spans
        total_slots = sum(span.num_slots for span in spans)
        used_slots = sum(len(span.slots) - len(span.freelist) for span in spans)
        
        return {
            'size_class': size_class,
            'spans': len(spans),
            'total_slots': total_slots,
            'used_slots': used_slots,
            'occupancy': (used_slots / total_slots * 100) if total_slots > 0 else 0
        }


###############################################################################
#                              JS MODULE
###############################################################################

class JSSprayGenerator:
    """Generates JavaScript code for heap spraying."""
    
    def __init__(self):
        self.templates = {
            'array_buffer': 'const {name} = new ArrayBuffer({size});',
            'uint8_array': 'const {name} = new Uint8Array({size});',
            'array': 'const {name} = new Array({length}).fill({value});',
            'object': 'const {name} = {{{props}}};',
            'string': 'const {name} = "{content}";',
        }
    
    def generate_array_buffer(self, name, size, count=1):
        template = self.templates['array_buffer']
        code_lines = []
        if count == 1:
            return template.format(name=name, size=size)
        
        for i in range(count):
            code_lines.append(template.format(name=f"{name}_{i}", size=size))
        return '\n'.join(code_lines)
    
    def generate_typed_array(self, name, size, count=1, array_type='Uint8Array'):
        template = f'const {{name}} = new {array_type}({{size}});'
        code_lines = []
        if count == 1:
            return template.format(name=name, size=size)
        
        for i in range(count):
            code_lines.append(template.format(name=f"{name}_{i}", size=size))
        return '\n'.join(code_lines)
    
    def generate_object_spray(self, name, properties, count=1):
        props_str = ', '.join([f'"{k}": {v}' for k, v in properties.items()])
        template = self.templates['object']
        code_lines = []
        if count == 1:
            return template.format(name=name, props=props_str)
        
        for i in range(count):
            code_lines.append(template.format(name=f"{name}_{i}", props=props_str))
        return '\n'.join(code_lines)
    
    def generate_string_spray(self, name, length, character='A', count=1):
        content = character * length
        template = self.templates['string']
        code_lines = []
        if count == 1:
            return template.format(name=name, content=content)
        
        for i in range(count):
            code_lines.append(template.format(name=f"{name}_{i}", content=content))
        return '\n'.join(code_lines)
    
    def generate_spray_array(self, name, object_template, length):
        return f'''
        const {name} = [];
        for (let i = 0; i < {length}; i++) {{
            {name}.push({object_template});
        }}
        '''.strip()
    
    def generate_hole_filler(self, target_size, count=100):
        if target_size <= 16:
            return self.generate_typed_array("fillers", 8, count, "Uint8Array")
        elif target_size <= 32:
            return self.generate_typed_array("fillers", 24, count, "Uint8Array")
        elif target_size <= 64:
            return self.generate_typed_array("fillers", 56, count, "Uint8Array")
        elif target_size <= 128:
            return self.generate_typed_array("fillers", 120, count, "Uint8Array")
        else:
            props = {f"p{i}": "1" for i in range((target_size - 16) // 8)}
            return self.generate_object_spray("fillers", props, count)
    
    def generate_defrag_code(self):
        return '''
        // Force garbage collection to consolidate free space
        function triggerGC() {
            const largeArray = new Array(1000000).fill(0);
            return largeArray;
        }
        
        function defragHeap() {
            // Allocate many small objects
            const smallObjs = [];
            for (let i = 0; i < 1000; i++) {
                smallObjs.push(new Uint8Array(32));
            }
            
            // Force garbage collection
            triggerGC();
            
            // Free some objects to create holes
            for (let i = 0; i < smallObjs.length; i += 2) {
                smallObjs[i] = null;
            }
            
            // Force garbage collection again
            triggerGC();
            
            // Fill holes with new objects
            for (let i = 0; i < 500; i++) {
                smallObjs.push(new Uint8Array(32));
            }
        }
        
        defragHeap();
        '''.strip()


###############################################################################
#                              ANALYSIS MODULE
###############################################################################

class FreelistAnalyzer:
    """Analyzes freelist structures and behavior."""
    
    def __init__(self, allocator):
        self.allocator = PartitionRoot()
        self.allocation_history = []  # (action, alloc_id, size, type)
    
    def record_allocation(self, alloc_id, size, object_type):
        self.allocation_history.append(('alloc', alloc_id, size, object_type))
    
    def record_free(self, alloc_id):
        self.allocation_history.append(('free', alloc_id, None, None))
    
    def get_freelist_state(self):
        result = {}
        for size_class, bucket in self.allocator.buckets.items():
            if not bucket.slot_spans:
                continue
            spans_data = []
            for i, span in enumerate(bucket.slot_spans):
                spans_data.append({
                    'span_idx': i,
                    'total_slots': span.num_slots,
                    'free_slots': len(span.freelist),
                    'occupancy': span.occupancy()
                })
            result[size_class] = {
                'spans': spans_data,
                'active_span': bucket.active_slot_span_idx
            }
        return result
    
    def analyze_reuse_pattern(self, size_class):
        test_allocator = type(self.allocator)()
        test_allocs = {}
        reuse_data = []
        
        for action, alloc_id, size, obj_type in self.allocation_history:
            if action == 'alloc':
                info = self.allocator.get_allocation_info(alloc_id)
                if info and info['size_class'] == size_class:
                    test_id = test_allocator.allocate(size, obj_type)
                    test_allocs[test_id] = alloc_id
                    test_info = test_allocator.get_allocation_info(test_id)
                    reuse_data.append({
                        'action': 'alloc',
                        'orig_id': alloc_id,
                        'test_id': test_id,
                        'span': test_info['span_idx'],
                        'slot': test_info['slot_idx']
                    })
            elif action == 'free':
                test_id = None
                for tid, aid in list(test_allocs.items()):
                    if aid == alloc_id:
                        test_id = tid
                        break
                if test_id:
                    info = test_allocator.get_allocation_info(test_id)
                    reuse_data.append({
                        'action': 'free',
                        'orig_id': alloc_id,
                        'test_id': test_id,
                        'span': info['span_idx'],
                        'slot': info['slot_idx']
                    })
                    test_allocator.free(test_id)
                    del test_allocs[test_id]
        
        return reuse_data
    
    def find_reuse_candidates(self, target_alloc_id):
        target_info = self.allocator.get_allocation_info(target_alloc_id)
        if not target_info:
            return []
        
        size_class = target_info['size_class']
        span_idx = target_info['span_idx']
        slot_idx = target_info['slot_idx']
        
        reuse_pattern = self.analyze_reuse_pattern(size_class)
        candidates = []
        target_freed = False
        
        for entry in reuse_pattern:
            if entry['action'] == 'free' and entry['orig_id'] == target_alloc_id:
                target_freed = True
                continue
            if target_freed and entry['action'] == 'alloc' \
               and entry['span'] == span_idx and entry['slot'] == slot_idx:
                candidates.append(entry['orig_id'])
        
        return candidates


class ObjectClassifier:
    """Classifies JavaScript objects based on exploitation potential."""
    
    CLASSIFICATIONS = {
        'ArrayBuffer': {
            'spray_candidate': True,
            'dangerous': False,
            'control_fields': ['byteLength'],
            'description': 'Good for controlled memory layout.'
        },
        'Uint8Array': {
            'spray_candidate': True,
            'dangerous': False,
            'control_fields': ['byteLength', 'byteOffset'],
            'description': 'Allows read/write to memory.'
        },
        'String': {
            'spray_candidate': True,
            'dangerous': False,
            'control_fields': ['length'],
            'description': 'Controlled memory layout, though less direct R/W.'
        },
        'Object': {
            'spray_candidate': True,
            'dangerous': False,
            'control_fields': [],
            'description': 'Generic JS object.'
        },
        'JSFunction': {
            'spray_candidate': False,
            'dangerous': True,
            'control_fields': ['code_entry', 'jitcode'],
            'description': 'Contains function pointers for potential hijack.'
        },
        'Node': {
            'spray_candidate': False,
            'dangerous': True,
            'control_fields': ['vtable'],
            'description': 'DOM node with a vtable pointer.'
        }
    }
    
    def __init__(self):
        self.custom_classifications = {}
    
    def add_custom_classification(self, object_type, classification):
        self.custom_classifications[object_type] = classification
    
    def classify(self, object_type):
        if object_type in self.custom_classifications:
            return self.custom_classifications[object_type]
        if object_type in self.CLASSIFICATIONS:
            return self.CLASSIFICATIONS[object_type]
        return {
            'spray_candidate': False,
            'dangerous': False,
            'control_fields': [],
            'description': 'Unknown object type.'
        }
    
    def get_spray_candidates(self, size_range=None):
        candidates = []
        for obj_type, info in self.CLASSIFICATIONS.items():
            if info['spray_candidate']:
                candidates.append(obj_type)
        for obj_type, info in self.custom_classifications.items():
            if info['spray_candidate']:
                candidates.append(obj_type)
        return candidates
    
    def get_dangerous_objects(self):
        dangerous = []
        for obj_type, info in self.CLASSIFICATIONS.items():
            if info['dangerous']:
                dangerous.append(obj_type)
        for obj_type, info in self.custom_classifications.items():
            if info['dangerous']:
                dangerous.append(obj_type)
        return dangerous


###############################################################################
#                              STRATEGIES MODULE
###############################################################################

class GroomingStrategy:
    """Represents a heap grooming strategy."""
    
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.steps = []
    
    def add_step(self, action, params=None):
        self.steps.append({
            'action': action,
            'params': params or {}
        })
    
    def to_js(self, js_generator):
        code_lines = [f'// {self.name}', f'// {self.description}', '']
        
        for step in self.steps:
            action = step['action']
            params = step['params']
            
            if action == 'spray_array_buffer':
                code = js_generator.generate_array_buffer(
                    params.get('name', 'buf'),
                    params.get('size', 64),
                    params.get('count', 1)
                )
            elif action == 'spray_typed_array':
                code = js_generator.generate_typed_array(
                    params.get('name', 'arr'),
                    params.get('size', 64),
                    params.get('count', 1),
                    params.get('array_type', 'Uint8Array')
                )
            elif action == 'spray_object':
                code = js_generator.generate_object_spray(
                    params.get('name', 'obj'),
                    params.get('properties', {'a': 1}),
                    params.get('count', 1)
                )
            elif action == 'spray_string':
                code = js_generator.generate_string_spray(
                    params.get('name', 'str'),
                    params.get('length', 16),
                    params.get('character', 'A'),
                    params.get('count', 1)
                )
            elif action == 'defrag':
                code = js_generator.generate_defrag_code()
            elif action == 'hole_filler':
                code = js_generator.generate_hole_filler(
                    params.get('target_size', 64),
                    params.get('count', 100)
                )
            elif action == 'custom':
                code = params.get('code', '// Custom code here')
            else:
                code = f'// Unknown action: {action}'
            
            code_lines.append(code)
            code_lines.append('')
        
        return '\n'.join(code_lines)


class StrategyGenerator:
    """Generates heap grooming strategies based on analysis."""
    
    def __init__(self, allocator, js_generator, object_classifier):
        self.allocator = allocator
        self.js_generator = js_generator
        self.object_classifier = object_classifier
    
    def generate_strategy_for_target(self, target_object_type, target_size=None):
        obj_info = self.object_classifier.classify(target_object_type)
        
        if target_size is None:
            if target_object_type == 'ArrayBuffer':
                target_size = 64
            elif target_object_type == 'JSFunction':
                target_size = 48
            elif target_object_type == 'Node':
                target_size = 96
            else:
                target_size = 32
        
        size_class = self.allocator.get_size_class(target_size)
        
        strategy = GroomingStrategy(
            f"Grooming for {target_object_type}",
            f"Strategy to place {target_object_type} objects ~{target_size} bytes in controlled positions."
        )
        
        # Step 1: Defragment the heap
        strategy.add_step('defrag')
        
        # Step 2: Spray filler objects
        strategy.add_step('hole_filler', {
            'target_size': size_class,
            'count': 200
        })
        
        # Step 3: Spray the target objects
        if target_object_type == 'ArrayBuffer':
            strategy.add_step('spray_array_buffer', {
                'name': 'targetBuf',
                'size': target_size,
                'count': 10
            })
        elif target_object_type == 'String':
            strategy.add_step('spray_string', {
                'name': 'targetStr',
                'length': target_size - 16,
                'count': 10
            })
        elif target_object_type == 'Uint8Array':
            strategy.add_step('spray_typed_array', {
                'name': 'targetArr',
                'size': target_size - 24,
                'count': 10,
                'array_type': 'Uint8Array'
            })
        else:
            props = {f"p{i}": "1" for i in range((target_size - 16) // 8)}
            strategy.add_step('spray_object', {
                'name': 'targetObj',
                'properties': props,
                'count': 10
            })
        
        # Step 4: Custom manipulation code
        strategy.add_step('custom', {
            'code': f'''
            // Access and manipulate target objects
            console.log("Target objects created");
            
            // Force GC to observe allocation stability
            function gc() {{
                for (let i = 0; i < 10; i++) {{
                    let a = new ArrayBuffer(1024 * 1024 * 10);
                }}
            }}
            gc();
            console.log("Garbage collection triggered");
            '''
        })
        
        return strategy
    
    def generate_strategy_for_bug(self, bug_type, affected_size, overflow_size=None):
        strategy = None
        if bug_type == 'uaf':
            strategy = self._generate_uaf_strategy(affected_size)
        elif bug_type == 'overflow':
            strategy = self._generate_overflow_strategy(affected_size, overflow_size)
        elif bug_type == 'double_free':
            strategy = self._generate_double_free_strategy(affected_size)
        else:
            strategy = GroomingStrategy(
                f"Generic Exploitation Strategy",
                f"Generic strategy for objects ~{affected_size} bytes."
            )
            strategy.add_step('defrag')
            strategy.add_step('hole_filler', {'target_size': affected_size, 'count': 100})
        return strategy
    
    def _generate_uaf_strategy(self, affected_size):
        size_class = self.allocator.get_size_class(affected_size)
        spray_candidates = self.object_classifier.get_spray_candidates()
        best_candidate = 'ArrayBuffer'
        if 'ArrayBuffer' in spray_candidates:
            best_candidate = 'ArrayBuffer'
        elif 'Uint8Array' in spray_candidates:
            best_candidate = 'Uint8Array'
        
        strategy = GroomingStrategy(
            "Use-After-Free Exploitation",
            f"Strategy for UAF bug with objects ~{affected_size} bytes."
        )
        
        strategy.add_step('defrag')
        
        # Step 1: Create vulnerable objects
        strategy.add_step('custom', {
            'code': f'''
            // Create vulnerable objects
            const vulnerableObjects = [];
            for (let i = 0; i < 20; i++) {{
                vulnerableObjects.push(new ArrayBuffer({affected_size}));
            }}
            console.log("Created vulnerable objects");
            '''
        })
        
        # Step 2: Trigger free
        strategy.add_step('custom', {
            'code': f'''
            // Free the vulnerable objects (simulate UAF)
            for (let i = 0; i < vulnerableObjects.length; i++) {{
                vulnerableObjects[i] = null;
            }}
            console.log("Triggered free of vulnerable objects");
            '''
        })
        
        # Step 3: Spray replacement objects
        if best_candidate == 'ArrayBuffer':
            strategy.add_step('spray_array_buffer', {
                'name': 'replacementBuf',
                'size': affected_size - 8,
                'count': 30
            })
        else:
            strategy.add_step('spray_typed_array', {
                'name': 'replacementArr',
                'size': affected_size - 24,
                'count': 30
            })
        
        # Step 4: Access dangling pointer
        strategy.add_step('custom', {
            'code': f'''
            // Access replaced objects
            console.log("Accessing replaced objects through dangling pointers");
            '''
        })
        
        return strategy
    
    def _generate_overflow_strategy(self, affected_size, overflow_size=8):
        overflow_size = overflow_size or 8
        size_class = self.allocator.get_size_class(affected_size)
        dangerous_objects = self.object_classifier.get_dangerous_objects()
        target_object = 'JSFunction' if 'JSFunction' in dangerous_objects else (
            dangerous_objects[0] if dangerous_objects else 'ArrayBuffer'
        )
        
        strategy = GroomingStrategy(
            "Heap Overflow Exploitation",
            f"Overflow of ~{overflow_size} bytes against objects ~{affected_size} bytes."
        )
        
        strategy.add_step('defrag')
        
        # Step 1: Create vulnerable object
        strategy.add_step('custom', {
            'code': f'''
            const vulnerableBuffer = new ArrayBuffer({affected_size});
            const vulnerableView = new Uint8Array(vulnerableBuffer);
            console.log("Created vulnerable object");
            '''
        })
        
        # Step 2: Place target objects
        if target_object == 'JSFunction':
            strategy.add_step('custom', {
                'code': f'''
                const targetFunctions = [];
                for (let i = 0; i < 50; i++) {{
                    targetFunctions.push(function() {{ return i; }});
                }}
                console.log("Placed target JSFunctions");
                '''
            })
        elif target_object == 'ArrayBuffer':
            strategy.add_step('spray_array_buffer', {
                'name': 'targetBuffer',
                'size': 64,
                'count': 50
            })
        else:
            strategy.add_step('custom', {
                'code': f'''
                const targetObjects = [];
                for (let i = 0; i < 50; i++) {{
                    targetObjects.push({{ 
                        value: i,
                        important: 0x12345678
                    }});
                }}
                console.log("Placed target objects");
                '''
            })
        
        # Step 3: Trigger overflow
        strategy.add_step('custom', {
            'code': f'''
            function triggerOverflow() {{
                const overflowData = new Uint8Array({overflow_size});
                for (let i = 0; i < overflowData.length; i++) {{
                    overflowData[i] = 0x41 + (i % 26);
                }}
                console.log("Overflow data prepared:", overflowData);
                // In a real exploit, you'd write beyond vulnerableView's bounds
            }}
            triggerOverflow();
            '''
        })
        
        return strategy
    
    def _generate_double_free_strategy(self, affected_size):
        size_class = self.allocator.get_size_class(affected_size)
        
        strategy = GroomingStrategy(
            "Double Free Exploitation",
            f"Double free bug on objects ~{affected_size} bytes."
        )
        
        strategy.add_step('defrag')
        
        # Step 1: Create object
        strategy.add_step('custom', {
            'code': f'''
            let vulnerableObject = new ArrayBuffer({affected_size});
            console.log("Created vulnerable object");
            '''
        })
        
        # Step 2: Spray
        strategy.add_step('hole_filler', {
            'target_size': affected_size,
            'count': 50
        })
        
        # Step 3: Trigger double free
        strategy.add_step('custom', {
            'code': f'''
            vulnerableObject = null;
            function gc() {{
                for (let i = 0; i < 10; i++) {{
                    let a = new ArrayBuffer(1024*1024*10);
                }}
            }}
            gc();
            console.log("Simulated double free (second free not actually performed here)");
            '''
        })
        
        # Step 4: Allocate after double free
        strategy.add_step('spray_array_buffer', {
            'name': 'exploitBuf',
            'size': affected_size - 8,
            'count': 20
        })
        
        return strategy


###############################################################################
#                              VISUALIZATION MODULE
###############################################################################

class TimelineVisualizer:
    """Visualizes the timeline of heap operations."""
    
    def __init__(self, allocator):
        self.allocator = allocator
        self.events = []  # (timestamp, event_type, details)
        self.snapshots = []  # (timestamp, allocator_state)
    
    def record_event(self, event_type, details=None):
        timestamp = len(self.events)
        self.events.append((timestamp, event_type, details or {}))
    
    def take_snapshot(self):
        timestamp = len(self.snapshots)
        
        bucket_stats = {}
        for size_class in self.allocator.buckets:
            stats = self.allocator.get_bucket_stats(size_class)
            if stats and stats['spans'] > 0:
                bucket_stats[size_class] = stats
        
        allocations = self.allocator.allocations
        object_counts = {}
        for alloc_id in allocations:
            info = self.allocator.get_allocation_info(alloc_id)
            if info:
                obj_type = info['object_type']
                if obj_type not in object_counts:
                    object_counts[obj_type] = 0
                object_counts[obj_type] += 1
        
        snapshot = {
            'timestamp': timestamp,
            'bucket_stats': bucket_stats,
            'object_counts': object_counts,
            'total_allocations': len(allocations)
        }
        
        self.snapshots.append((timestamp, snapshot))
        return timestamp
    
    def plot_timeline(self, figsize=(12, 8)):
        if not self.snapshots:
            print("No snapshots available.")
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        timestamps = [t for t, _ in self.snapshots]
        total_allocs = [s['total_allocations'] for _, s in self.snapshots]
        
        all_size_classes = set()
        for _, snapshot in self.snapshots:
            all_size_classes.update(snapshot['bucket_stats'].keys())
        size_classes = sorted(all_size_classes)
        
        data = np.zeros((len(size_classes), len(timestamps)))
        for i, (_, snapshot) in enumerate(self.snapshots):
            for j, sc in enumerate(size_classes):
                if sc in snapshot['bucket_stats']:
                    data[j, i] = snapshot['bucket_stats'][sc]['used_slots']
        
        ax1.plot(timestamps, total_allocs, linewidth=2)
        ax1.set_xlabel('Timeline')
        ax1.set_ylabel('Total Allocations')
        ax1.set_title('Allocation Count Over Time')
        ax1.grid(True)
        
        ax2.stackplot(timestamps, data, labels=[f'{s} bytes' for s in size_classes], alpha=0.7)
        ax2.set_xlabel('Timeline')
        ax2.set_ylabel('Used Slots')
        ax2.set_title('Slot Usage by Size Class')
        ax2.grid(True)
        ax2.legend(loc='upper left', fontsize='small')
        
        plt.tight_layout()
        return fig
    
    def plot_bucket_occupancy(self, size_class, figsize=(10, 6)):
        if not self.snapshots:
            print("No snapshots available.")
            return None
        
        data = []
        timestamps = []
        for t, snapshot in self.snapshots:
            if size_class in snapshot['bucket_stats']:
                stats = snapshot['bucket_stats'][size_class]
                timestamps.append(t)
                data.append(stats['occupancy'])
        
        if not data:
            print(f"No data for size class {size_class}.")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(timestamps, data, linewidth=2)
        ax.set_xlabel('Timeline')
        ax.set_ylabel('Occupancy (%)')
        ax.set_title(f'Bucket Occupancy for Size Class {size_class} bytes')
        ax.grid(True)
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        return fig
    
    def compare_snapshots(self, snapshot1_id, snapshot2_id):
        if snapshot1_id >= len(self.snapshots) or snapshot2_id >= len(self.snapshots):
            return "Invalid snapshot IDs."
        
        _, snapshot1 = self.snapshots[snapshot1_id]
        _, snapshot2 = self.snapshots[snapshot2_id]
        
        differences = {
            'total_allocations': {
                'before': snapshot1['total_allocations'],
                'after': snapshot2['total_allocations'],
                'diff': snapshot2['total_allocations'] - snapshot1['total_allocations']
            },
            'buckets': {},
            'objects': {}
        }
        
        all_buckets = set(list(snapshot1['bucket_stats'].keys()) + list(snapshot2['bucket_stats'].keys()))
        for bucket in all_buckets:
            before = snapshot1['bucket_stats'].get(bucket, {'used_slots': 0, 'total_slots': 0})
            after = snapshot2['bucket_stats'].get(bucket, {'used_slots': 0, 'total_slots': 0})
            differences['buckets'][bucket] = {
                'used_slots_before': before['used_slots'],
                'used_slots_after':  after['used_slots'],
                'diff': after['used_slots'] - before['used_slots'],
                'spans_before': before.get('spans', 0),
                'spans_after':  after.get('spans', 0),
                'spans_diff': after.get('spans', 0) - before.get('spans', 0)
            }
        
        all_objects = set(list(snapshot1['object_counts'].keys()) + list(snapshot2['object_counts'].keys()))
        for obj_type in all_objects:
            before = snapshot1['object_counts'].get(obj_type, 0)
            after = snapshot2['object_counts'].get(obj_type, 0)
            differences['objects'][obj_type] = {
                'before': before,
                'after': after,
                'diff': after - before
            }
        
        return differences


class FreelistVisualizer:
    """Visualizes the state of freelists."""
    
    def __init__(self, allocator):
        self.allocator = allocator
    
    def print_freelist_state(self, size_class=None):
        if size_class:
            self._print_size_class(size_class)
        else:
            for sc in sorted(self.allocator.buckets.keys()):
                bucket = self.allocator.buckets[sc]
                if bucket.slot_spans:
                    self._print_size_class(sc)
    
    def _print_size_class(self, size_class):
        if size_class not in self.allocator.buckets:
            print(f"Size class {size_class} does not exist.")
            return
        
        bucket = self.allocator.buckets[size_class]
        print(f"\n== Size Class: {size_class} bytes ==")
        print(f"Active slot span: {bucket.active_slot_span_idx}")
        print(f"Total slot spans: {len(bucket.slot_spans)}")
        
        for i, span in enumerate(bucket.slot_spans):
            active_marker = " (ACTIVE)" if i == bucket.active_slot_span_idx else ""
            free_pct = len(span.freelist) / span.num_slots * 100
            alloc_pct = 100 - free_pct
            print(f"\nSpan {i}{active_marker}: {span.num_slots} slots, "
                  f"{len(span.freelist)} free ({free_pct:.1f}%), "
                  f"{span.num_slots - len(span.freelist)} allocated ({alloc_pct:.1f}%)")
            self._visualize_span(span)
    
    def _visualize_span(self, span, width=80):
        num_slots = span.num_slots
        if num_slots <= 0:
            return
        
        slots_per_line = min(width, num_slots)
        lines_needed = (num_slots + slots_per_line - 1) // slots_per_line
        
        for line in range(lines_needed):
            start_idx = line * slots_per_line
            end_idx = min(start_idx + slots_per_line, num_slots)
            line_str = ""
            for i in range(start_idx, end_idx):
                line_str += "□" if span.slots[i] is None else "■"
            print(line_str)
    
    def visualize_bucket(self, size_class):
        if size_class not in self.allocator.buckets:
            print(f"Size class {size_class} does not exist.")
            return
        
        bucket = self.allocator.buckets[size_class]
        print(f"\n=== Detailed View of Size Class: {size_class} bytes ===")
        for i, span in enumerate(bucket.slot_spans):
            active_marker = " (ACTIVE)" if i == bucket.active_slot_span_idx else ""
            print(f"\nSpan {i}{active_marker}:")
            for j in range(span.num_slots):
                slot_status = "FREE" if span.slots[j] is None else f"ALLOCATED ({span.slots[j]})"
                in_freelist = j in span.freelist
                freelist_idx = span.freelist.index(j) if in_freelist else None
                extra = f", Freelist pos: {freelist_idx}" if in_freelist else ""
                print(f"  Slot {j}: {slot_status}{extra}")


class DiffVisualizer:
    """Visualizes differences between snapshots."""
    
    def __init__(self, timeline_visualizer):
        self.timeline_visualizer = timeline_visualizer
    
    def print_diff(self, snapshot1_id, snapshot2_id):
        diff = self.timeline_visualizer.compare_snapshots(snapshot1_id, snapshot2_id)
        if isinstance(diff, str):
            print(diff)
            return
        
        print("\n=== Snapshot Comparison ===")
        print(f"Snapshot {snapshot1_id} vs Snapshot {snapshot2_id}")
        
        total_diff = diff['total_allocations']
        print(f"\nTotal Allocations: {total_diff['before']} -> {total_diff['after']} "
              f"({total_diff['diff']:+d})")
        
        print("\nBucket Changes:")
        for bucket, bucket_diff in sorted(diff['buckets'].items()):
            used_diff = bucket_diff['diff']
            spans_diff = bucket_diff['spans_diff']
            if used_diff != 0 or spans_diff != 0:
                print(f"  Size Class {bucket} bytes:")
                print(f"    Used Slots: {bucket_diff['used_slots_before']} -> "
                      f"{bucket_diff['used_slots_after']} ({used_diff:+d})")
                print(f"    Spans: {bucket_diff['spans_before']} -> "
                      f"{bucket_diff['spans_after']} ({spans_diff:+d})")
        
        print("\nObject Type Changes:")
        for obj_type, obj_diff in sorted(diff['objects'].items()):
            if obj_diff['diff'] != 0:
                print(f"  {obj_type}: {obj_diff['before']} -> {obj_diff['after']} "
                      f"({obj_diff['diff']:+d})")
    
    def plot_diff(self, snapshot1_id, snapshot2_id, figsize=(12, 6)):
        diff = self.timeline_visualizer.compare_snapshots(snapshot1_id, snapshot2_id)
        if isinstance(diff, str):
            print(diff)
            return None
        
        buckets = []
        slot_diffs = []
        for bucket, bd in sorted(diff['buckets'].items()):
            if bd['diff'] != 0:
                buckets.append(str(bucket))
                slot_diffs.append(bd['diff'])
        
        if not buckets:
            print("No bucket differences to plot.")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        colors = ['g' if d >= 0 else 'r' for d in slot_diffs]
        y_pos = np.arange(len(buckets))
        
        ax.barh(y_pos, slot_diffs, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(buckets)
        ax.invert_yaxis()
        ax.set_xlabel('Change in Used Slots')
        ax.set_title(f'Bucket Changes: Snapshot {snapshot1_id} to {snapshot2_id}')
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        return fig


###############################################################################
#                              TARGET (BUG) MODULE
###############################################################################

class BugSimulator:
    """Simulates various heap-related bugs for analysis."""
    
    def __init__(self, allocator, analyzer, visualizer):
        self.allocator = allocator
        self.analyzer = analyzer
        self.visualizer = visualizer
    
    def simulate_use_after_free(self, target_alloc_id):
        target_info = self.allocator.get_allocation_info(target_alloc_id)
        if not target_info:
            return {
                'success': False,
                'error': f"Allocation ID {target_alloc_id} not found."
            }
        
        snapshot_before = self.visualizer.take_snapshot()
        self.allocator.free(target_alloc_id)
        self.analyzer.record_free(target_alloc_id)
        
        snapshot_after_free = self.visualizer.take_snapshot()
        reuse_candidates = self.analyzer.find_reuse_candidates(target_alloc_id)
        
        return {
            'success': True,
            'target_info': target_info,
            'snapshot_before': snapshot_before,
            'snapshot_after_free': snapshot_after_free,
            'reuse_candidates': reuse_candidates,
            'bug_type': 'use_after_free',
            'message': f"Simulated UAF for {target_info['object_type']} in size class {target_info['size_class']}"
        }
    
    def simulate_buffer_overflow(self, source_alloc_id, overflow_size=8):
        source_info = self.allocator.get_allocation_info(source_alloc_id)
        if not source_info:
            return {
                'success': False,
                'error': f"Allocation ID {source_alloc_id} not found."
            }
        
        snapshot_before = self.visualizer.take_snapshot()
        
        size_class = source_info['size_class']
        span_idx = source_info['span_idx']
        slot_idx = source_info['slot_idx']
        bucket = self.allocator.buckets[size_class]
        span = bucket.slot_spans[span_idx]
        
        adjacent_slot_idx = slot_idx + 1
        affected_alloc_id = None
        if adjacent_slot_idx < span.num_slots and span.slots[adjacent_slot_idx] is not None:
            for alloc_id, alloc_data in self.allocator.allocations.items():
                a_size_class, a_span_idx, a_slot_idx = alloc_data
                if (a_size_class == size_class and 
                    a_span_idx == span_idx and 
                    a_slot_idx == adjacent_slot_idx):
                    affected_alloc_id = alloc_id
                    break
        
        affected_info = None
        if affected_alloc_id:
            affected_info = self.allocator.get_allocation_info(affected_alloc_id)
        
        return {
            'success': True,
            'source_info': source_info,
            'affected_id': affected_alloc_id,
            'affected_info': affected_info,
            'overflow_size': overflow_size,
            'snapshot_before': snapshot_before,
            'bug_type': 'buffer_overflow',
            'message': f"Simulated {overflow_size}-byte overflow from {source_info['object_type']}"
        }
    
    def simulate_double_free(self, target_alloc_id):
        target_info = self.allocator.get_allocation_info(target_alloc_id)
        if not target_info:
            return {
                'success': False,
                'error': f"Allocation ID {target_alloc_id} not found."
            }
        
        snapshot_before = self.visualizer.take_snapshot()
        self.allocator.free(target_alloc_id)
        self.analyzer.record_free(target_alloc_id)
        
        snapshot_after_free = self.visualizer.take_snapshot()
        
        return {
            'success': True,
            'target_info': target_info,
            'snapshot_before': snapshot_before,
            'snapshot_after_free': snapshot_after_free,
            'bug_type': 'double_free',
            'message': f"Simulated double free for {target_info['object_type']} in size class {target_info['size_class']}"
        }


###############################################################################
#                              TARGET (RECOMMENDATIONS) MODULE
###############################################################################

class ExploitRecommender:
    """Recommends exploitation strategies based on bug simulation."""
    
    def __init__(self, js_generator, strategy_generator, object_classifier):
        self.js_generator = js_generator
        self.strategy_generator = strategy_generator
        self.object_classifier = object_classifier
    
    def recommend_for_bug(self, bug_simulation):
        if not bug_simulation['success']:
            return {
                'success': False,
                'error': bug_simulation.get('error', 'Bug simulation failed.')
            }
        
        bug_type = bug_simulation['bug_type']
        if bug_type == 'use_after_free':
            return self._recommend_for_uaf(bug_simulation)
        elif bug_type == 'buffer_overflow':
            return self._recommend_for_overflow(bug_simulation)
        elif bug_type == 'double_free':
            return self._recommend_for_double_free(bug_simulation)
        else:
            return {
                'success': False,
                'error': f"Unknown bug type: {bug_type}"
            }
    
    def _recommend_for_uaf(self, bug_simulation):
        target_info = bug_simulation['target_info']
        size_class = target_info['size_class']
        object_type = target_info['object_type']
        
        spray_candidates = self.object_classifier.get_spray_candidates()
        dangerous_objects = self.object_classifier.get_dangerous_objects()
        
        strategy = self.strategy_generator.generate_strategy_for_bug('uaf', size_class)
        js_code = strategy.to_js(self.js_generator)
        
        return {
            'bug_type': 'use_after_free',
            'target_object': {
                'type': object_type,
                'size_class': size_class
            },
            'spray_candidates': spray_candidates,
            'dangerous_replacements': dangerous_objects,
            'strategy': {
                'name': strategy.name,
                'description': strategy.description,
                'js_code': js_code
            },
            'technique_recommendations': [
                "After freeing the target object, spray the heap with objects of the same size.",
                "Use ArrayBuffers for controlled content or JSFunctions for code execution.",
                "Ensure replacement objects share the same size class for reuse.",
                "Map out offset if targeting a specific field."
            ]
        }
    
    def _recommend_for_overflow(self, bug_simulation):
        source_info = bug_simulation['source_info']
        size_class = source_info['size_class']
        object_type = source_info['object_type']
        overflow_size = bug_simulation['overflow_size']
        
        affected_info = bug_simulation.get('affected_info')
        strategy = self.strategy_generator.generate_strategy_for_bug('overflow', size_class, overflow_size)
        js_code = strategy.to_js(self.js_generator)
        
        corruptible_fields = []
        if affected_info:
            classification = self.object_classifier.classify(affected_info['object_type'])
            corruptible_fields = classification.get('control_fields', [])
        
        rec = {
            'bug_type': 'buffer_overflow',
            'source_object': {
                'type': object_type,
                'size_class': size_class
            },
            'overflow_size': overflow_size,
            'affected_object': None,
            'strategy': {
                'name': strategy.name,
                'description': strategy.description,
                'js_code': js_code
            },
            'technique_recommendations': [
                f"{overflow_size}-byte overflow can corrupt the adjacent object.",
                "Place an object with exploitable fields right after the vulnerable buffer.",
                "Control the layout for consistent positioning.",
                "JSFunctions are prime targets for pointer corruption."
            ]
        }
        if affected_info:
            rec['affected_object'] = {
                'type': affected_info['object_type'],
                'size_class': affected_info['size_class'],
                'corruptible_fields': corruptible_fields
            }
        
        return rec
    
    def _recommend_for_double_free(self, bug_simulation):
        target_info = bug_simulation['target_info']
        size_class = target_info['size_class']
        object_type = target_info['object_type']
        
        strategy = self.strategy_generator.generate_strategy_for_bug('double_free', size_class)
        js_code = strategy.to_js(self.js_generator)
        
        return {
            'bug_type': 'double_free',
            'target_object': {
                'type': object_type,
                'size_class': size_class,
            },
            'strategy': {
                'name': strategy.name,
                'description': strategy.description,
                'js_code': js_code
            },
            'technique_recommendations': [
                "A double free can corrupt the allocator freelist.",
                "Allocate objects to overlap in memory after the corruption.",
                "Use different object types to see if content overlaps occur.",
                "Monitor for crashes or anomalies that indicate success."
            ]
        }


###############################################################################
#                              MAIN / CLI MODULE
###############################################################################

class HeapKit:
    """Main class for the heap grooming toolkit."""
    
    def __init__(self):
        self.allocator = PartitionAlloc()
        self.js_generator = JSSprayGenerator()
        self.analyzer = FreelistAnalyzer(self.allocator)
        self.object_classifier = ObjectClassifier()
        self.strategy_generator = StrategyGenerator(self.allocator, self.js_generator, self.object_classifier)
        self.timeline_visualizer = TimelineVisualizer(self.allocator)
        self.freelist_visualizer = FreelistVisualizer(self.allocator)
        self.diff_visualizer = DiffVisualizer(self.timeline_visualizer)
        self.bug_simulator = BugSimulator(self.allocator, self.analyzer, self.timeline_visualizer)
        self.recommender = ExploitRecommender(self.js_generator, self.strategy_generator, self.object_classifier)
        
        self.next_allocation_id = 1
        self.allocations = {}  # Maps user-friendly IDs -> internal IDs
    
    def allocate(self, size, object_type):
        alloc_id = self.allocator.allocate(size, object_type)
        custom_id = f"alloc_{self.next_allocation_id}"
        self.next_allocation_id += 1
        self.allocations[custom_id] = alloc_id
        self.analyzer.record_allocation(alloc_id, size, object_type)
        self.timeline_visualizer.record_event('allocate', {
            'id': custom_id,
            'alloc_id': alloc_id,
            'size': size,
            'type': object_type
        })
        return custom_id
    
    def free(self, custom_id):
        if custom_id not in self.allocations:
            print(f"Error: Allocation {custom_id} not found.")
            return False
        alloc_id = self.allocations[custom_id]
        self.allocator.free(alloc_id)
        self.analyzer.record_free(alloc_id)
        self.timeline_visualizer.record_event('free', {
            'id': custom_id,
            'alloc_id': alloc_id
        })
        del self.allocations[custom_id]
        return True
    
    def generate_spray_code(self, object_type, size, count=100):
        if object_type == 'ArrayBuffer':
            return self.js_generator.generate_array_buffer('spray', size, count)
        elif object_type in ('Uint8Array', 'Uint16Array', 'Uint32Array'):
            return self.js_generator.generate_typed_array('spray', size, count, object_type)
        elif object_type == 'String':
            return self.js_generator.generate_string_spray('spray', size, 'A', count)
        else:
            props = {f"p{i}": "1" for i in range(max(1, (size - 16) // 8))}
            return self.js_generator.generate_object_spray('spray', props, count)
    
    def generate_strategy(self, target_object_type, target_size=None):
        strategy = self.strategy_generator.generate_strategy_for_target(target_object_type, target_size)
        return strategy.to_js(self.js_generator)
    
    def simulate_bug(self, bug_type, target_id, **kwargs):
        if target_id not in self.allocations:
            print(f"Error: Allocation {target_id} not found.")
            return None
        
        alloc_id = self.allocations[target_id]
        if bug_type == 'uaf':
            return self.bug_simulator.simulate_use_after_free(alloc_id)
        elif bug_type == 'overflow':
            overflow_size = kwargs.get('overflow_size', 8)
            return self.bug_simulator.simulate_buffer_overflow(alloc_id, overflow_size)
        elif bug_type == 'double_free':
            return self.bug_simulator.simulate_double_free(alloc_id)
        else:
            print(f"Error: Unknown bug type '{bug_type}'.")
            return None
    
    def get_recommendations(self, bug_simulation):
        return self.recommender.recommend_for_bug(bug_simulation)
    
    def take_snapshot(self):
        snapshot_id = self.timeline_visualizer.take_snapshot()
        print(f"Snapshot {snapshot_id} taken.")
        return snapshot_id
    
    def compare_snapshots(self, snapshot1_id, snapshot2_id):
        self.diff_visualizer.print_diff(snapshot1_id, snapshot2_id)
    
    def visualize_timeline(self):
        fig = self.timeline_visualizer.plot_timeline()
        if fig:
            plt.show()
    
    def visualize_bucket(self, size_class):
        fig = self.timeline_visualizer.plot_bucket_occupancy(size_class)
        if fig:
            plt.show()
    
    def print_freelist_state(self, size_class=None):
        self.freelist_visualizer.print_freelist_state(size_class)
    
    def detailed_bucket_view(self, size_class):
        self.freelist_visualizer.visualize_bucket(size_class)
    
    def run_demo(self):
        print("=== HeapKit Demonstration ===\n")
        
        print("Step 1: Allocating objects...")
        allocations = []
        for i in range(5):
            alloc_id = self.allocate(64, 'ArrayBuffer')
            allocations.append(alloc_id)
            print(f"  Allocated {alloc_id}: ArrayBuffer of size 64")
        
        for i in range(3):
            alloc_id = self.allocate(128, 'Uint8Array')
            allocations.append(alloc_id)
            print(f"  Allocated {alloc_id}: Uint8Array of size 128")
        
        target_id = self.allocate(256, 'JSFunction')
        print(f"  Allocated {target_id}: JSFunction of size 256 (target)")
        
        print("\nTaking snapshot after allocations...")
        snapshot1 = self.take_snapshot()
        
        print("\nStep 2: Freeing some objects...")
        for i in range(0, len(allocations), 2):
            free_id = allocations[i]
            self.free(free_id)
            print(f"  Freed {free_id}")
        
        print("\nTaking snapshot after frees...")
        snapshot2 = self.take_snapshot()
        
        print("\nStep 3: Comparing snapshots...")
        self.compare_snapshots(snapshot1, snapshot2)
        
        print("\nStep 4: Current freelist state...")
        self.print_freelist_state()
        
        print("\nStep 5: Generating heap spray strategy for ArrayBuffer...")
        strategy_code = self.generate_strategy('ArrayBuffer', 64)
        print(strategy_code)
        
        print("\nStep 6: Simulating a use-after-free bug on the target...")
        bug_sim = self.simulate_bug('uaf', target_id)
        if bug_sim and bug_sim['success']:
            print(f"  {bug_sim['message']}")
            print("\nStep 7: Exploitation recommendations...")
            recs = self.get_recommendations(bug_sim)
            if not recs.get('error'):
                print(f"  Bug Type: {recs['bug_type']}")
                print(f"  Target Object: {recs['target_object']['type']} (Size Class: {recs['target_object']['size_class']})")
                print("\n  Spray Candidates:")
                for c in recs['spray_candidates']:
                    print(f"    - {c}")
                print("\n  Dangerous Replacement Objects:")
                for d in recs['dangerous_replacements']:
                    print(f"    - {d}")
                print("\n  Technique Recommendations:")
                for tr in recs['technique_recommendations']:
                    print(f"    - {tr}")
        
        print("\nDemonstration complete!")


def main():
    parser = argparse.ArgumentParser(description='HeapKit: Browser Heap Exploitation Toolkit')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    spray_parser = subparsers.add_parser('spray', help='Generate heap spray code')
    spray_parser.add_argument('--type', required=True, help='Object type to spray')
    spray_parser.add_argument('--size', type=int, required=True, help='Object size in bytes')
    spray_parser.add_argument('--count', type=int, default=100, help='Number of objects to create')
    
    strategy_parser = subparsers.add_parser('strategy', help='Generate a grooming strategy')
    strategy_parser.add_argument('--target', required=True, help='Target object type')
    strategy_parser.add_argument('--size', type=int, help='Target object size')
    
    simulate_parser = subparsers.add_parser('simulate', help='Simulate a heap exploitation scenario')
    simulate_parser.add_argument('--bug', required=True, choices=['uaf', 'overflow', 'double_free'], 
                                 help='Bug type to simulate')
    
    demo_parser = subparsers.add_parser('demo', help='Run a demonstration of the toolkit')
    
    args = parser.parse_args()
    heapkit = HeapKit()
    
    if args.command == 'spray':
        js_code = heapkit.generate_spray_code(args.type, args.size, args.count)
        print(js_code)
    
    elif args.command == 'strategy':
        js_code = heapkit.generate_strategy(args.target, args.size)
        print(js_code)
    
    elif args.command == 'simulate':
        print("Running simulation as part of the demonstration (requires allocated objects).")
        heapkit.run_demo()
    
    elif args.command == 'demo' or not args.command:
        heapkit.run_demo()
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
