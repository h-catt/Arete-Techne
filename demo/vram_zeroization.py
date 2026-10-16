
import torch
import gc

def display_vram(label):
    """Display GPU memory usage in MiB."""
    torch.cuda.synchronize()

    allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)

    print(f"\n--- {label} ---")
    print(f"PyTorch allocated: {allocated:.2f} MiB")
    print(f"PyTorch reserved:  {reserved:.2f} MiB")
    print(f"GPU memory used:   {(total_bytes - free_bytes) / (1024 ** 2):.2f} MiB")
    print(f"GPU memory free:   {free_bytes / (1024 ** 2):.2f} MiB")


def test_vram_zeroization(size_mb=1):
    if not torch.cuda.is_available():
        print("ERROR: CUDA unavailable. Select a GPU runtime.")
        return False

    device = torch.device("cuda:0")
    size_bytes = size_mb * 1024 * 1024

    # BEFORE: baseline memory
    display_vram("BEFORE TEST — baseline")

    buffer = None

    try:
        # Allocate GPU memory
        buffer = torch.empty(
            size_bytes,
            dtype=torch.uint8,
            device=device
        )

        # Write and verify recognizable pattern
        buffer.fill_(0xA5)
        torch.cuda.synchronize()

        pattern_verified = torch.all(buffer == 0xA5).item()

        if not pattern_verified:
            print("FAIL: test pattern verification failed.")
            return False

        print(f"\nPASS: {size_mb} MiB buffer allocated.")
        print("PASS: recognizable pattern verified.")

        # DURING: buffer allocated
        display_vram("DURING TEST — pattern allocated")

        # Overwrite all bytes with zeros
        buffer.zero_()
        torch.cuda.synchronize()

        nonzero_bytes = torch.count_nonzero(buffer).item()

        print("\n--- ZEROIZATION CHECK ---")
        if nonzero_bytes == 0:
            print("PASS: all bytes in the test buffer read as zero.")
        else:
            print(f"FAIL: {nonzero_bytes} nonzero bytes remain.")
            return False

        # Memory usage while zeroed buffer still exists
        display_vram("AFTER OVERWRITE — buffer still allocated")

    finally:
        # Release buffer reference
        if buffer is not None:
            del buffer

        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # AFTER: memory following cleanup
        display_vram("AFTER TEST — buffer released")

    return nonzero_bytes == 0


if __name__ == "__main__":
    test_vram_zeroization(size_mb=1)
