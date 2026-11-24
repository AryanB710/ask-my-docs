# LumenQueue Overview

LumenQueue is a managed message queue service for sending messages between services without running your own brokers.

## Core concepts

A **queue** stores messages until a consumer receives and deletes them. Each message is delivered to exactly one consumer in a consumer group.

A **topic** fans out every published message to all subscribed queues. Use topics when several independent services need the same event.

A **consumer group** is a set of workers that share the messages of one queue. Adding workers to a consumer group increases throughput; LumenQueue balances messages across them automatically.

A **message** has a body (up to the payload limit of your API version), optional attributes, and a system-assigned message ID that begins with the prefix `msg_`.

## Delivery guarantees

Standard queues provide at-least-once delivery. A message may occasionally be delivered more than once, so consumers should be idempotent.

FIFO queues provide exactly-once processing and strict ordering within a message group. FIFO queue names must end with the suffix `.fifo`. FIFO queues support up to 300 messages per second per message group.

## Regions

LumenQueue is available in five regions: `us-east`, `us-west`, `eu-central`, `ap-south` (Mumbai) and `ap-southeast` (Singapore). Data never leaves the region where the queue was created unless you configure cross-region replication.

Cross-region replication is available only on the Enterprise tier.

## Tiers

LumenQueue has three tiers: Free, Standard and Enterprise. See the Billing page for prices and the SLA page for uptime commitments.
