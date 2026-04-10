# Third-Party Model References

BasicSTFM includes adapters for the following spatio-temporal foundation model
families:

- OpenCity: https://github.com/HKUDS/OpenCity
- FactoST: https://github.com/CityMind-Lab/FactoST
- UniST: https://github.com/tsinghua-fib-lab/UniST

`basicstfm.third_party.opencity.original_opencity` is a verbatim copy of the
OpenCity model file inspected from the MIT-licensed OpenCity repository. The
upstream license is MIT:

Copyright (c) 2026 Data Intelligence Lab@HKU

The registered BasicSTFM adapters under `basicstfm.models.foundation` are
framework-native implementations that preserve the public STFM design ideas
needed by this project: graph-temporal encoding for OpenCity, universal temporal
pretraining plus factorized ST adaptation for FactoST, and masked pretraining
plus prompt tuning for UniST.

The inspected FactoST and UniST repositories did not include a LICENSE file in
their public checkouts. For that reason, BasicSTFM does not vendor substantial
FactoST or UniST source files; it provides clean adapter implementations and
keeps the official repository links here for attribution and comparison.
