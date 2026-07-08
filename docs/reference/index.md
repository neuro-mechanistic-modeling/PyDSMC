# API Reference

This section documents PyDSMC's public API. Everything below is importable directly from the
top-level `pydsmc` package:

```python
from pydsmc import (
    Evaluator,
    Property,
    create_predefined_property,
    create_custom_property,
    get_predefined_properties,
    create_eval_envs,
    jsons_to_df,
)
```

## Overview

| Symbol | Purpose |
| --- | --- |
| [`Evaluator`](evaluator.md) | Runs Monte-Carlo rollouts of a policy and estimates each registered property until convergence or a resource limit. |
| [`Property`](property.md#pydsmc.Property) | A trajectory-based quantity to estimate, together with its statistical method. |
| [`create_predefined_property`](property.md#pydsmc.create_predefined_property) | Build a ready-made property (e.g. `return`, `episode_length`). |
| [`create_custom_property`](property.md#pydsmc.create_custom_property) | Build a property from your own `check_fn`. |
| [`get_predefined_properties`](property.md#pydsmc.get_predefined_properties) | List the available predefined property IDs. |
| [`create_eval_envs`](utils.md#pydsmc.create_eval_envs) | Helper to construct vectorized evaluation environments. |
| [`jsons_to_df`](utils.md#pydsmc.jsons_to_df) | Aggregate JSON-Lines run logs into a pandas `DataFrame`. |

See [Usage](../usage.md) for a conceptual walkthrough.
