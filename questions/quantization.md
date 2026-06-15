# Quantization — open questions

**Status:** parked — revisit after exploring other parts of the problem space.

**Why parked:** Scalar (f32→int8) and product quantization cut the bytes streamed
per query, which directly targets the memory-bandwidth bound measured in
`history/001` and `002`. Deferred deliberately — want to test other approaches
to the within-cell scan first before trading recall for bytes.

## Questions

_(to add)_
