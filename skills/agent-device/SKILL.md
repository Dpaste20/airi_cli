---
name: agent-device
description: Mobile device automation CLI for AI agents. Use when the user needs to interact with iOS simulators/physical devices or Android emulators/devices, including navigating apps, filling forms, tapping buttons, taking screenshots, extracting UI data, testing mobile apps, or automating any device interaction. Triggers include requests to "open an app", "tap a button on iPhone", "fill a form on Android", "take a screenshot on simulator", "scrape data from a mobile screen", "test this mobile app", "automate device actions", "push a notification", or any task requiring programmatic iOS/Android UI interaction.
---

# Mobile Automation with agent-device

`agent-device` is the mobile-native counterpart to `agent-browser` — built for iOS simulators, iOS physical devices, Android emulators, and Android devices.

## Core Workflow

Every device automation follows this pattern:

1. **Boot** (if no device ready): `agent-device boot --platform ios`
2. **Open app**: `agent-device open SampleApp --platform ios`
3. **Snapshot** to get element refs: `agent-device snapshot -i`
4. **Interact** using refs: `agent-device click @e2`
5. **Re-snapshot** after any UI change before reusing refs

```bash
agent-device open SampleApp --platform ios
agent-device snapshot -i
# Output:
# @e1 [heading] "Sample App"
# @e2 [button] "Settings"
# @e3 [text-field] "Search"

agent-device click @e2
agent-device snapshot -i        # MUST re-snapshot after navigation
agent-device fill @e3 "query"
agent-device close
```

## Essential Commands

```bash
# Boot & navigation
agent-device boot --platform ios
agent-device boot --platform android --device Pixel_9_Pro_XL
agent-device boot --platform android --device Pixel_9_Pro_XL --headless
agent-device open SampleApp --platform ios        # open app
agent-device open Settings                        # switch app in current session
agent-device open "https://example.com" --platform ios     # open URL in browser
agent-device open MyApp "myapp://screen/to" --platform ios # open deep link
agent-device close                                # close current app
agent-device back                                 # hardware back (Android)
agent-device home                                 # go to home screen
agent-device app-switcher                         # open app switcher

# Snapshot
agent-device snapshot -i                 # Interactive elements only (recommended)
agent-device snapshot -c                 # Compact (less noise)
agent-device snapshot -d 3               # Limit tree depth to 3
agent-device snapshot -s "Contacts"      # Scope to label/identifier
agent-device snapshot -i -c -d 5         # Combine options
agent-device diff snapshot               # Structural diff vs previous baseline

# Interactions (use @refs from snapshot)
agent-device click @e1                   # Tap element
agent-device fill @e2 "text"             # Clear then type (verifies on Android)
agent-device type "text"                 # Type into focused field without clearing
agent-device focus @e3                   # Focus element
agent-device scroll down 0.5             # Scroll down half screen
agent-device scroll up 0.3              # Scroll up 30%
agent-device scrollintoview "Sign in"   # Scroll until text is visible
agent-device scrollintoview @e42        # Scroll ref into view
agent-device swipe 540 1500 540 500 120  # Swipe from (x1,y1) to (x2,y2) durationMs
agent-device longpress 300 500 800       # Long press at coords for 800ms
agent-device press 300 500               # Tap at coordinates
agent-device pinch 2.0                   # Zoom in 2x (iOS simulator only)
agent-device pinch 0.5 200 400           # Zoom out at coords (iOS simulator only)

# Get information
agent-device get text @e1               # Get element text
agent-device get attrs @e1              # Get element attributes

# Screenshot & recording
agent-device screenshot                 # Auto-named screenshot
agent-device screenshot page.png        # Save to specific path
agent-device record start               # Start screen recording
agent-device record start session.mp4   # Record to explicit path
agent-device record start session.mp4 --fps 30  # With FPS cap (physical iOS)
agent-device record stop                # Stop active recording
```

## Ref Lifecycle (Critical)

Refs (`@e1`, `@e2`, etc.) are invalidated whenever the page/screen changes. **Always re-snapshot after:**
- Tapping links or buttons that navigate
- Form submissions
- Dynamic content loading (modals, drawers)
- `scrollintoview` (use ref-mode only for geometry scrolling, then re-snapshot)

```bash
agent-device click @e5          # Navigates to new screen
agent-device snapshot -i        # MUST re-snapshot
agent-device click @e1          # Use new refs
```

## Semantic Selectors (Alternative to Refs)

When refs are unavailable or unreliable, use `find`:

```bash
agent-device find "Sign In" click
agent-device find text "Sign In" click
agent-device find label "Email" fill "user@example.com"
agent-device find value "Search" type "query"
agent-device find role button click
agent-device find id "com.example:id/login" click
agent-device find "Submit" wait 3000    # Wait up to 3s for element to appear
```

`find ... click --json` returns structured match data:
```json
{ "ref": "@e3", "locator": "any", "query": "Sign In", "x": 195, "y": 422 }
```

## Sessions (Parallel & Isolated)

```bash
# Default session
agent-device open Settings --platform ios
agent-device session list

# Named session (for parallel work)
agent-device open Contacts --platform ios --session my-session
agent-device snapshot -i --session my-session
agent-device close --session my-session

# Shutdown simulator/emulator on close (good for CI)
agent-device close --shutdown
```

Sessions keep device state and snapshot baselines consistent across commands. Use `--session <name>` to run multiple device sessions in parallel.

## Fast Batching

When you already know a sequence of actions, batch them in one daemon request to reduce overhead:

```bash
# From a file
agent-device batch \
  --platform ios \
  --steps-file /tmp/batch-steps.json \
  --json

# Inline for small payloads
agent-device batch --steps '[{"command":"open","positionals":["settings"]},{"command":"wait","positionals":["100"]}]'
```

**Step payload format:**
```json
[
  { "command": "open",  "positionals": ["settings"] },
  { "command": "wait",  "positionals": ["label=\"Privacy & Security\"", "3000"] },
  { "command": "click", "positionals": ["label=\"Privacy & Security\""] },
  { "command": "get",   "positionals": ["text", "label=\"Tracking\""] }
]
```

**Batch best practices:**
- Batch only one related screen flow at a time (5–20 steps)
- Insert `wait`/`is exists` guards after mutating steps (`click`, `fill`, `swipe`)
- Treat all refs as stale after UI changes — re-snapshot in the next batch phase
- On failure, replan from `details.step` and `details.partialResults`
- Prefer `--steps-file` over inline JSON for longer sequences

## Settings Helpers

```bash
# Network & system (iOS simulator + Android)
agent-device settings wifi on|off
agent-device settings airplane on|off
agent-device settings location on|off
agent-device settings appearance light|dark|toggle

# Biometrics (iOS simulator only)
agent-device settings faceid match|nonmatch|enroll|unenroll
agent-device settings touchid match|nonmatch|enroll|unenroll

# Android fingerprint
agent-device settings fingerprint match|nonmatch

# App permissions (scoped to active session app)
agent-device settings permission grant camera
agent-device settings permission deny microphone
agent-device settings permission grant photos limited   # iOS only: full|limited
agent-device settings permission reset notifications
```

Note: iOS `settings` commands are **simulator-only**.

## Push Notification Simulation

```bash
# iOS (simulator-only, APNs-style JSON)
agent-device push com.example.app ./payload.apns --platform ios
agent-device push com.example.app '{"aps":{"alert":"Welcome","badge":1}}' --platform ios

# Android (adb broadcast)
agent-device push com.example.app '{"action":"com.example.app.PUSH","extras":{"title":"Welcome","unread":3}}' --platform android
```

## App Management

```bash
# Install / reinstall
agent-device install com.example.app ./build/app.apk        # in-place install
agent-device reinstall com.example.app ./build/app.apk       # uninstall + fresh install
agent-device install-from-source https://example.com/app.apk --platform android

# App state & listings
agent-device appstate                           # foreground app (Android: live; iOS: session-scoped)
agent-device apps --platform ios               # list installed apps
agent-device apps --platform android --all     # include system apps
```

## Clipboard

```bash
agent-device clipboard read                    # Read clipboard text
agent-device clipboard write "https://example.com"  # Write clipboard text
agent-device clipboard write ""               # Clear clipboard
```
Supported on Android emulator/device and iOS simulator. iOS physical devices return `UNSUPPORTED_OPERATION`.

## Performance Metrics

```bash
agent-device perf --json         # Session startup timing (run open first)
agent-device metrics --json      # Alias for perf
```

## App Logs (Debugging)

```bash
agent-device logs path           # Get log file path
agent-device logs start          # Start streaming app logs to file
agent-device logs stop           # Stop streaming
agent-device logs clear          # Truncate log file
agent-device logs clear --restart   # Stop, clear, and restart streaming

agent-device logs mark "before submit"   # Insert timeline marker

# Parse HTTP requests from log
agent-device network dump 25             # Recent 25 HTTP entries (method/url/status)
agent-device network dump 25 all        # Include headers/body (truncated)

# Efficient grepping (keeps token use low)
agent-device logs path
grep -n "Error\|Exception\|Fatal" ~/.agent-device/sessions/default/app.log
tail -50 ~/.agent-device/sessions/default/app.log
```

## TV Targets

```bash
agent-device open YouTube --platform android --target tv
agent-device open Settings --platform ios --target tv
agent-device screenshot apple-tv.png --platform ios --target tv
agent-device apps --platform android --target tv
```

## JSON Output

For programmatic parsing in scripts:

```bash
agent-device snapshot --json
agent-device get text @e1 --json
agent-device find "Sign In" click --json
agent-device batch --steps-file /tmp/steps.json --json
```

Note: Default text output is more compact and preferred for AI agents.

## Efficient Snapshot Usage

- Default to `snapshot -i` in agent loops — interactive-only is fastest
- Add `-s "<label>"` to scope results to a specific screen section
- Add `-d <depth>` when only upper hierarchy layers are needed
- Use `diff snapshot` between mutations to validate changes with lower output volume
- Re-snapshot after **any** UI mutation before reusing refs

## Common Patterns

### Form Submission

```bash
agent-device open MyApp --platform android
agent-device snapshot -i
agent-device fill @e1 "user@example.com"
agent-device fill @e2 "password123"
agent-device find role button click --name "Submit"
agent-device wait --load 2000
agent-device snapshot -i
```

### Login Reset (Fresh State)

```bash
agent-device reinstall com.example.app ./build/app.apk --platform android
agent-device open com.example.app --platform android
agent-device snapshot -i
```

### Parallel Sessions

```bash
agent-device open AppA --platform ios --session session-a
agent-device open AppB --platform android --session session-b

agent-device snapshot -i --session session-a
agent-device snapshot -i --session session-b

agent-device close --session session-a
agent-device close --session session-b
```

### Debug Loop

```bash
agent-device logs clear --restart
agent-device open MyApp --platform ios
# reproduce the bug...
agent-device logs path
grep -n "Error" ~/.agent-device/sessions/default/app.log
agent-device screenshot bug.png
```

## iOS Physical Device Prerequisites

- Xcode + `xcrun devicectl` available
- Paired device with Developer Mode enabled
- Use Automatic Signing in Xcode, or set env overrides:
  - `AGENT_DEVICE_IOS_TEAM_ID`
  - `AGENT_DEVICE_IOS_SIGNING_IDENTITY` (optional)
  - `AGENT_DEVICE_IOS_PROVISIONING_PROFILE`
- First-run XCTest build may be slow — increase timeout if needed:
  - `AGENT_DEVICE_DAEMON_TIMEOUT_MS=120000` (default: `90000`)
