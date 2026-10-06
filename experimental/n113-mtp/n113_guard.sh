#!/bin/bash
# usage: n110_guard.sh "<since: YYYY-mm-dd HH:MM:SS>" <label>   exit 0 = clean, 2 = TRIPPED
since="$1"; label="${2:-check}"
h=$(curl -s -m 5 -o /dev/null -w "%{http_code}" localhost:30141/health)
k=$(timeout 90 journalctl -k --since "$since" --no-pager 2>/dev/null)
bad=$(printf '%s\n' "$k" | grep -E -i "Completion-Wait loop timed out|pciehp.*(link down|removal|card not present|surprise)|event severity: fatal|Uncorrected \(Fatal\)|severity=Uncorrected \(Fatal\)" | head -5)
warn=$(printf '%s\n' "$k" | grep -E -i "event severity: (recoverable|uncorrected)|Uncorrected \(Non-Fatal\)|xe 0000:84:00.0.*(error|fail|reset|hang)" | head -3)
nhw=$(printf '%s\n' "$k" | grep -c "Hardware Error")
nsev=$(printf '%s\n' "$k" | grep -i "event severity" | sed -E 's/.*event severity: *//' | sort | uniq -c | tr '\n' ' ')
dst=$(ps -eo stat,pid,comm | awk '$1 ~ /^D/ && $3 ~ /khugepaged|kcompactd/')
echo "[$(date '+%F %T')] guard $label since='$since' health30141=$h hwerr_lines=$nhw severities={$nsev} dstate='${dst}'"
[ -n "$warn" ] && echo "WARN: $warn"
if [ "$h" != 200 ] || [ -n "$bad" ] || [ -n "$dst" ]; then echo "TRIPPED: health=$h bad=[$bad] dstate=[$dst]"; exit 2; fi
exit 0
