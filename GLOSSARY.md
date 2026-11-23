# Glossary

## Prefill (P)

The stage that consumes a request prompt and produces the key/value state and
final prompt hidden state needed by Decode.

## Decode (D)

The stage that owns coda execution, first-token sampling and subsequent token
generation after a transfer is received.

## Transfer Generation (`tid`)

A unique identity for one handoff of one request. It separates a current
handoff from an older handoff that used the same external request ID.

## Logical Payload

The key/value state and final prompt hidden state required by Decode. Protocol
metadata and transport framing are outside the logical payload.

## Received

A transfer state in which Decode has received every declared payload chunk and
the destination device writes are safe to read.

## Activated

A transfer state in which the received request has entered Decode scheduling.
