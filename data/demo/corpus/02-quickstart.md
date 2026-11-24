# Quickstart

This guide takes you from zero to your first published message in about five minutes.

## Install the CLI

Install the `lq` command-line tool with pip:

```
pip install lumenqueue-cli
```

The CLI requires Python 3.9 or later. Check the installed version with `lq --version`.

## Authenticate

Create an API key in the console under **Settings → API keys**. API keys start with the prefix `lqk_`.

Export the key as an environment variable so the CLI and SDKs can find it:

```
export LUMENQUEUE_API_KEY=lqk_your_key_here
```

API keys do not expire automatically, but we recommend rotating them every 90 days. You can have at most 10 active API keys per project.

## Create a queue

```
lq queues create orders --region ap-south
```

Queue names must be 3 to 80 characters long and may contain lowercase letters, digits, hyphens and underscores.

## Publish and receive

```
lq publish orders --body '{"order_id": 42}'
lq receive orders --max 10
```

After a consumer processes a message it must delete it with `lq delete`, otherwise the message becomes visible again when its visibility timeout expires.

## Next steps

- Read about rate limits before load testing.
- Configure a dead-letter queue for messages that repeatedly fail.
