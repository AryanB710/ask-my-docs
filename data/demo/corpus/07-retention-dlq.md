# Message Retention, Visibility Timeout and Dead-Letter Queues

## Retention period

Messages that are not deleted are kept for the queue's retention period and then removed automatically.

- Free tier: retention is fixed at 1 day.
- Standard tier: retention is configurable from 1 minute to 7 days. The default is 4 days.
- Enterprise tier: retention is configurable from 1 minute to 14 days. The default is 4 days.

Changing the retention period applies to messages already in the queue as well as new ones.

## Visibility timeout

When a consumer receives a message, the message is hidden from other consumers for the visibility timeout. The default visibility timeout is 30 seconds and the maximum is 12 hours.

If the consumer does not delete the message before the timeout expires, the message becomes visible again and can be received by another consumer. Long-running consumers can extend the timeout with the `ChangeVisibility` call.

## Dead-letter queues

A dead-letter queue (DLQ) is a normal queue that receives messages that could not be processed. You attach a DLQ to a source queue with a redrive policy.

The redrive policy sets the **max receive count**: the number of times a message can be received without being deleted before it is moved to the DLQ. The default max receive count is 5, and it can be set between 1 and 1,000.

A DLQ must be in the same region and the same project as its source queue. For FIFO queues, the DLQ must also be a FIFO queue.

## Redriving messages

After fixing the bug that caused failures, you can move messages from the DLQ back to the source queue with `lq redrive start <dlq-name>`. Redrive moves up to 500 messages per second.
