# API Limits and Quotas

LumenQueue enforces limits per project. Limits differ between the legacy v1 API and the current v2 API.

## Rate limits

### v2 API (current)

The v2 API allows 3,000 publish requests per second per project and 5,000 receive requests per second per project. Bursts of up to twice the sustained rate are allowed for 10 seconds.

### v1 API (legacy)

The legacy v1 API allows 500 publish requests per second per project and 1,000 receive requests per second per project. The v1 API does not allow bursts.

When you exceed a rate limit, the API returns HTTP status 429 with the error code `LQ-1029`. The response includes a `Retry-After` header in seconds.

## Payload size

The maximum message payload on the v2 API is 1 MB. The legacy v1 API is limited to 256 KB per message. Requests above the limit fail with HTTP 413 and error code `LQ-1013`.

For larger payloads, store the data in object storage and publish a reference to it.

## Batching

A single batch publish call can contain up to 100 messages on v2, and up to 10 messages on v1. The total size of a batch cannot exceed 5 MB.

## Queue quotas

Each project can have up to 1,000 queues on the Standard tier and 10,000 queues on the Enterprise tier. The Free tier is limited to 5 queues.

Quota increases can be requested from the console under **Support → Quota request**. Requests are usually reviewed within two business days.
