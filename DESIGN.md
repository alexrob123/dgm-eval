# Design choices

Design choices for the codebase.

## Metrics

### PR curves

**Common grid on x-axis**  
Right after computation, PR curves are interpolated to match a grid of evenly spaced points on \[0, 1\].   
It allows meaningful computation of aggregated curves (over labels in the pipeline, then over runs in post processing of results).