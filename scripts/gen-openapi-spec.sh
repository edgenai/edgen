# =====================================================================================================================
# Generates the Edgen OpenAPI Spec for edgend
# -------------------------------------------
# Input:
# - edgend source code
#
# Output:
# - OpenAPI Spec as yaml
# - OpenAPI Spec as json
# --------------------------------------------------------------------------------------------------------------------
# (c) Binedge 2023
# =====================================================================================================================
cargo run --bin edgend -- -o > openapi-docs/edgen-api-spec.yaml
swagger-cli bundle -o openapi-docs/edgen-api-spec.json openapi-docs/edgen-api-spec.yaml

