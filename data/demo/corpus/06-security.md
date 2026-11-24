# Security and Compliance

## Encryption

All messages are encrypted at rest with AES-256 and in transit with TLS 1.2 or later. Connections using TLS 1.0 or 1.1 are rejected.

By default LumenQueue manages encryption keys and rotates them automatically every 365 days. Enterprise customers can bring their own customer-managed keys (CMK); with CMK, key rotation is controlled by the customer.

## Access control

Access is controlled with project roles: Owner, Admin, Developer and Viewer. Only Owners can delete a project or change billing settings. Viewers can read queue metrics but cannot publish or receive messages.

Single sign-on (SSO) with SAML 2.0 is available on the Enterprise tier only.

## Network restrictions

You can restrict API access to a list of allowed IP ranges (IP allowlist). Each project supports up to 50 CIDR ranges in its allowlist. IP allowlists are available on Standard and Enterprise tiers.

## Audit logs

Every administrative action, such as creating a queue, rotating an API key or changing a role, is recorded in the audit log. Audit logs are retained for 90 days on the Standard tier and for 1 year on the Enterprise tier. Audit logs can be exported to object storage in JSON format.

## Compliance

LumenQueue is designed for workloads that must keep data inside a chosen region. Customers are responsible for not placing payment card numbers or other regulated data in message attributes, because attributes are visible in the console.
