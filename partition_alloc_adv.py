import math

# Constants for this model.
SUPERPAGE_SIZE = 2 * 1024 * 1024  # 2 MiB
PARTITION_PAGE_SIZE = 64 * 1024   # 64 KiB
PAGES_PER_SUPERPAGE = SUPERPAGE_SIZE // PARTITION_PAGE_SIZE
DIRECT_MAP_THRESHOLD = 512 * 1024  # For demonstration, 512 KiB.

class PartitionPage:
    """
    Represents a 64 KiB partition page, which can hold one or more SlotSpans.
    In real PartitionAlloc, typically one PartitionPage corresponds to one
    SlotSpan for a given bucket size. We'll simplify and keep one active slot
    span per PartitionPage.
    """
    def __init__(self, superpage_id, page_idx, size_class):
        self.superpage_id = superpage_id
        self.page_idx = page_idx
        self.size_class = size_class
        self.slot_span = None  # We'll create one slot span per page.
        self.is_active = True  # If the span is fully used or empty, we might toggle this.

class SlotSpan:
    """
    Represents a slot span within a PartitionPage. Contains a freelist of slots.
    """
    def __init__(self, size_class):
        self.size_class = size_class
        # How many slots fit in a 64 KiB partition page for this size class?
        # In real code, there's metadata overhead. We'll simplify.
        self.max_slots = (PARTITION_PAGE_SIZE // size_class)
        self.slots = [None] * self.max_slots  # None = free, otherwise object type
        self.freelist = list(range(self.max_slots))  # Basic integer freelist
        self.num_allocated = 0

    def allocate(self, object_type):
        if not self.freelist:
            return None
        slot_idx = self.freelist.pop(0)
        self.slots[slot_idx] = object_type
        self.num_allocated += 1
        return slot_idx

    def free(self, slot_idx):
        if self.slots[slot_idx] is None:
            raise ValueError("Double free detected (slot already free)!")
        self.slots[slot_idx] = None
        self.freelist.append(slot_idx)
        self.num_allocated -= 1

    def is_empty(self):
        return self.num_allocated == 0

    def is_full(self):
        return len(self.freelist) == 0

class Superpage:
    """
    Represents a 2 MiB superpage subdivided into 32 PartitionPages.
    We'll keep track of which pages are used or free.
    """
    def __init__(self, superpage_id):
        self.superpage_id = superpage_id
        self.pages = [None] * PAGES_PER_SUPERPAGE  # None means uninitialized page
        self.num_active_pages = 0

    def find_free_page_index(self):
        for i in range(PAGES_PER_SUPERPAGE):
            if self.pages[i] is None:
                return i
        return None

class DirectMapping:
    """
    Represents a direct-mapped allocation (for large allocations).
    In real code, this might involve OS-level page allocation. We'll just store metadata.
    """
    def __init__(self, size, object_type):
        self.size = size
        self.object_type = object_type

class Bucket:
    """
    Represents a bucket for a given size class. We store a list of "active" pages
    or partially filled slot spans. If all spans are full, we allocate a new one.
    """
    def __init__(self, size_class):
        self.size_class = size_class
        # We'll keep references to (superpage_id, page_idx) here
        self.active_pages = []  # List of (superpage_id, page_idx)

class PartitionRoot:
    """
    The main structure. Manages:
      - A dictionary of buckets keyed by size class
      - A list or dictionary of superpages
      - Direct mappings for large allocations
    """
    DEFAULT_SIZE_CLASSES = [
        8, 16, 32, 48, 64, 80, 96, 112, 128,
        144, 160, 176, 192, 208, 224, 240, 256,
        288, 320, 352, 384, 416, 448, 480, 512,
        576, 640, 704, 768, 832, 896, 960, 1024,
        1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048,
        2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096,
        # ... Add more if you want ...
    ]

    def __init__(self, size_classes=None):
        self.size_classes = size_classes or self.DEFAULT_SIZE_CLASSES
        self.buckets = {sc: Bucket(sc) for sc in self.size_classes}
        self.superpages = {}    # superpage_id -> Superpage
        self.direct_mapped = {} # alloc_id -> DirectMapping
        self.allocations = {}   # alloc_id -> Metadata about the allocation
        self.next_alloc_id = 1
        self.next_superpage_id = 1

    def get_size_class(self, size):
        """
        Find the nearest bucket size >= requested size.
        If none, we pick the largest bucket (or direct map).
        """
        for sc in self.size_classes:
            if size <= sc:
                return sc
        # If bigger than all known classes, we might direct map or just pick last class
        return self.size_classes[-1]

    def allocate(self, size, object_type):
        """
        Entry point for allocating an object of given size.
        """
        # Check direct map threshold
        if size > DIRECT_MAP_THRESHOLD:
            return self._allocate_direct_map(size, object_type)

        # Otherwise, bucket allocation
        sc = self.get_size_class(size)
        return self._allocate_bucket(sc, object_type)

    def _allocate_direct_map(self, size, object_type):
        alloc_id = self.next_alloc_id
        self.next_alloc_id += 1

        dm = DirectMapping(size, object_type)
        self.direct_mapped[alloc_id] = dm
        self.allocations[alloc_id] = {
            'type': 'direct_map',
            'size': size,
            'object_type': object_type
        }
        return alloc_id

    def _allocate_bucket(self, size_class, object_type):
        bucket = self.buckets[size_class]

        # Try to find an active page that has a free slot
        for (sup_id, page_idx) in bucket.active_pages:
            sp = self.superpages[sup_id]
            page = sp.pages[page_idx]
            slot_idx = page.slot_span.allocate(object_type)
            if slot_idx is not None:
                alloc_id = self.next_alloc_id
                self.next_alloc_id += 1
                self.allocations[alloc_id] = {
                    'type': 'bucket',
                    'superpage_id': sup_id,
                    'page_idx': page_idx,
                    'slot_idx': slot_idx,
                    'size_class': size_class,
                    'object_type': object_type
                }
                return alloc_id

        # If we got here, we need a new PartitionPage (and possibly a new Superpage).
        sup_id = self._ensure_superpage()
        sp = self.superpages[sup_id]
        page_idx = sp.find_free_page_index()
        if page_idx is None:
            # That means no free page in this superpage; let's create another superpage
            sup_id = self._ensure_superpage()
            sp = self.superpages[sup_id]
            page_idx = sp.find_free_page_index()

        # Initialize the page
        page = PartitionPage(sup_id, page_idx, size_class)
        page.slot_span = SlotSpan(size_class)
        sp.pages[page_idx] = page
        sp.num_active_pages += 1

        # Insert this page into the bucket's active list
        bucket.active_pages.append((sup_id, page_idx))

        # Now allocate
        slot_idx = page.slot_span.allocate(object_type)
        alloc_id = self.next_alloc_id
        self.next_alloc_id += 1
        self.allocations[alloc_id] = {
            'type': 'bucket',
            'superpage_id': sup_id,
            'page_idx': page_idx,
            'slot_idx': slot_idx,
            'size_class': size_class,
            'object_type': object_type
        }
        return alloc_id

    def _ensure_superpage(self):
        """
        Create a new superpage if none is fully free.
        Return the ID of a superpage that has at least one free PartitionPage.
        """
        # Try existing superpages first
        for sup_id, sp in self.superpages.items():
            idx = sp.find_free_page_index()
            if idx is not None:
                return sup_id

        # If none has a free page, create a new superpage
        sup_id = self.next_superpage_id
        self.next_superpage_id += 1
        sp = Superpage(sup_id)
        self.superpages[sup_id] = sp
        return sup_id

    def free(self, alloc_id):
        """
        Free the given allocation ID. If direct-mapped, remove from direct_mapped.
        If bucket-based, find the relevant slot span and free the slot.
        """
        if alloc_id not in self.allocations:
            raise ValueError(f"Allocation ID {alloc_id} not found.")

        info = self.allocations[alloc_id]
        if info['type'] == 'direct_map':
            # Just remove from direct_mapped
            del self.direct_mapped[alloc_id]
            del self.allocations[alloc_id]
            return

        if info['type'] == 'bucket':
            sup_id = info['superpage_id']
            page_idx = info['page_idx']
            slot_idx = info['slot_idx']
            sp = self.superpages[sup_id]
            page = sp.pages[page_idx]
            page.slot_span.free(slot_idx)
            # We could do additional logic if slot_span is empty or fully free
            # to release the page or remove from active lists, etc.
            del self.allocations[alloc_id]

    def get_allocation_info(self, alloc_id):
        """
        Return metadata about the allocation, if any.
        """
        return self.allocations.get(alloc_id, None)

    def get_bucket_stats(self, size_class):
        """
        Return stats about how many slot spans are active,
        how many pages, etc.
        """
        if size_class not in self.buckets:
            return None
        bucket = self.buckets[size_class]
        # Count how many pages are associated with this bucket, how many free vs used slots
        total_slots = 0
        used_slots = 0
        page_count = 0
        for (sup_id, page_idx) in bucket.active_pages:
            sp = self.superpages[sup_id]
            page = sp.pages[page_idx]
            if page and page.slot_span:
                total_slots += page.slot_span.max_slots
                used_slots += page.slot_span.num_allocated
                page_count += 1
        occupancy = (used_slots / total_slots * 100) if total_slots else 0
        return {
            'size_class': size_class,
            'active_pages': page_count,
            'total_slots': total_slots,
            'used_slots': used_slots,
            'occupancy': occupancy
        }
