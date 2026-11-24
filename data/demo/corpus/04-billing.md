# Billing and Pricing

LumenQueue bills per million API requests plus storage. All prices are in US dollars.

## Free tier

The Free tier includes 1 million requests per month and 1 GB of message storage at no cost. When a Free project exceeds its monthly request allowance, requests are throttled until the next billing cycle; you are never charged on the Free tier.

## Standard tier

The Standard tier costs $0.40 per million requests. Message storage costs $0.10 per GB-month. There is no monthly minimum.

## Enterprise tier

The Enterprise tier costs $0.30 per million requests with a monthly minimum commitment of $2,000. Enterprise includes cross-region replication, SSO and a dedicated support engineer.

## How requests are counted

Every API call counts as one request, including receive calls that return no messages. A batch publish of up to 100 messages counts as one request, which is why batching reduces cost.

Payloads are metered in 64 KB chunks: a 200 KB message counts as four requests.

## Invoices and payment

Invoices are issued on the first day of each month for the previous month's usage. Payment is due within 30 days. Accepted payment methods are credit card and bank transfer; bank transfer is available only for Enterprise customers.

If an invoice is unpaid 15 days after its due date, the project is suspended and publish requests are rejected until the balance is settled.

## Refunds

Prepaid Enterprise commitments are non-refundable. Billing disputes must be raised within 60 days of the invoice date.
