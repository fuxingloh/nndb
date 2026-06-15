# IVF — open questions

**Status:** parked — not building IVF in this engine.

**Why parked:** This engine is assumed to *be* one IVF cell/shard. The coarse
quantizer routes a query to a cell at a layer above us; our component does the
efficient search over that cell's vectors. So IVF-the-router is above us — our
job is the within-cell scan. Open questions about IVF *optimization* (how the
cell search should cooperate with the coarse quantizer) get logged here.

## Questions

_(to add)_
