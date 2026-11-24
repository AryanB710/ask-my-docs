# Troubleshooting and Error Codes

## Common error codes

| HTTP status | Error code | Meaning |
|---|---|---|
| 400 | LQ-1000 | The request body is malformed JSON. |
| 401 | LQ-1001 | The API key is missing or invalid. |
| 403 | LQ-1003 | The API key is valid but its role does not allow this action. |
| 403 | LQ-1033 | The request came from an IP address outside the project's allowlist. |
| 404 | LQ-1004 | The queue does not exist in this region. |
| 413 | LQ-1013 | The message payload exceeds the size limit. |
| 429 | LQ-1029 | The project exceeded its rate limit. |
| 503 | LQ-1503 | The service is temporarily unavailable; retry the request. |

Note that LQ-1003 and LQ-1033 both return HTTP 403. LQ-1003 is about permissions of the role, while LQ-1033 is about the network location of the caller.

## Retries and backoff

Retry requests that fail with 429 or 503. Use exponential backoff starting at 100 milliseconds, doubling on each attempt, with full jitter, up to a maximum of 20 seconds between attempts. The official SDKs retry up to 3 times by default.

Do not retry 400, 401, 403 or 413 errors; they will fail again until the request is fixed.

## Messages delivered twice

On standard queues, duplicate delivery is expected occasionally because of at-least-once delivery. It also happens when processing takes longer than the visibility timeout. Make consumers idempotent, for example by recording processed message IDs.

## Queue not found

If you get LQ-1004 for a queue you just created, check that the CLI or SDK is configured for the same region as the queue. Queues are regional and are not visible from other regions.
