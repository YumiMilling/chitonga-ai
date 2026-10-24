"""The WhatsApp adapter for the harness, PILOT MODE (24 Oct 2026, the Jesuit; Chris: "can we start using the actual
harness for the training with our teachers?").

Nothing in this package sends a message unless THREE things are true at the moment of the send: the live flag
(env ZAMAI_ASSISTANT_LIVE=1), Chris's go file (internal/chitonga_train/harness/PILOT_GO.json with his words), and no
stop (no PILOT_STOP file, no env ZAMAI_ASSISTANT_STOP). Simulation and --dry-run use a mocked sender. See PILOT_PLAN.md.
"""
