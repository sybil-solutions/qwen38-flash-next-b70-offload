#!/bin/bash
# N111 guard (since BOOT): kernel log of this boot must have no Completion-Wait timeouts, PCIe Link Down / Card not
# present, "reboot is needed", or non-corrected GHES Hardware Error records; /health 200; no D-state khugepaged/kcompactd;
# the B70 0000:84:00.0 must be bound to xe as renderD130. usage: guard_boot.sh <label>
lab=$1
h=$(curl -s -m 5 -o /dev/null -w "%{http_code}" localhost:30141/health)
echo "GUARD $lab $(date -Is) health=$h boot=$(uptime -s)"
log=$(timeout 20 journalctl -k -b --no-pager 2>/dev/null)
[ -n "$log" ] || { echo "KERNEL_LOG_UNREADABLE"; echo GUARD_FAIL; exit 1; }
bad1=$(echo "$log" | grep -E "Completion-Wait loop timed out|Link Down|Card not present|reboot is needed")
bad2=$(echo "$log" | awk 'match($0,/\{[0-9]+\}\[Hardware Error\]/){ id=substr($0,RSTART,RLENGTH);
   if ($0 ~ /event severity:/){ s=$0; sub(/.*event severity: */,"",s); sev[id]=s } seen[id]=1 }
   /Hardware Error/ && /[Ff]atal/ && !/corrected/ { print "FATAL: " $0 }
   END{ for (i in seen) if (sev[i]!="" && sev[i]!="corrected") print "NONCORR " i " sev=" sev[i] }')
rn=""; for p in /sys/class/drm/renderD*; do [ "$(basename $(readlink -f $p/device))" = 0000:84:00.0 ] && rn=$(basename $p); done
echo "b70_84_node=${rn:-absent}"
dst=$(ps -eo stat=,comm= | awk '$1 ~ /^D/ && $2 ~ /khugepaged|kcompactd/')
if [ -n "$bad1$bad2" ]; then echo "KERNEL_BAD:"; echo "$bad1" | tail -5; echo "$bad2" | tail -5; else echo "KERNEL_OK (since boot)"; fi
[ -n "$dst" ] && echo "DSTATE: $dst" || echo "DSTATE_OK"
[ "$h" = 200 ] && [ -z "$bad1$bad2" ] && [ -z "$dst" ] && [ "$rn" = renderD130 ] && echo GUARD_PASS || { echo GUARD_FAIL; exit 1; }
