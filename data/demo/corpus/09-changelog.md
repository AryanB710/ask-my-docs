# Changelog and Deprecations

## v1 API deprecation

The v1 API is deprecated. It will stop accepting requests on 31 March 2027. All new projects created after 1 June 2026 can use only the v2 API.

To migrate, change the base URL from `/v1/` to `/v2/` and update SDKs to version 4.0 or later. The v2 API renames the `ttl` field to `retention_seconds`.

## Release history

### SDK 4.2 (October 2026)

Added the `ChangeVisibility` helper to the Python and Go SDKs and support for the `ap-southeast` region.

### SDK 4.0 (June 2026)

First SDK release that targets the v2 API by default. Dropped support for Python 3.8.

### Console update (August 2026)

Added the quota request form under **Support → Quota request** and audit log export to object storage.

## Planned

FIFO throughput of 1,000 messages per second per message group is planned but has no release date yet.
