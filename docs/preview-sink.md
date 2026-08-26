# Preview sink protocol

`OutputStage` retains its recording path and forwards each completed packet to `PreviewSink`.
Preview publication uses a bounded `DropOldest` queue, so a preview consumer cannot delay
capture, pose processing, or disk recording. A `PreviewPacket` is a `shared_ptr<const Packet>`:
the packet and every GPU buffer owner therefore remain alive until the asynchronous transport
has completed its use of the frame.

## Shared memory

When legacy compatibility is enabled, the existing v1 mapping remains at the configured
destination. Version 2 is always published at `<destination>_v2`. Both mappings start with an
odd/even sequence lock followed by packet sequence and payload size. Readers must read the lock,
copy the payload, then re-read the lock; a packet is valid only if both values match and are even.

The v2 payload begins with little-endian magic `IRS2` (`0x32535249`) and protocol version `2`,
then payload size, packet sequence, frame count, and pose count. Each frame encodes camera ID,
frame sequence, dimensions, pixel format, stride, byte size, CUDA device ID, source timestamp,
and a CUDA IPC handle. Pose records currently contain `sourceSequence`. Fields are written
individually; readers must not interpret C++ object layouts as wire data.

Preview faults are recorded in transport health and are non-fatal. Start and stop are idempotent;
stop closes the queue and joins the consumer before releasing mappings.
