#!/bin/bash
# N111 copy of N109 guard + D-state khugepaged/kcompactd check
# usage: guard.sh <label> [since]
# Health 200; AER totals; kernel log: pciehp / Completion-Wait / any NON-corrected GHES record ([Hardware Error])
# mentioning 80:/81:/82:/83:/84: devices. GHES records with "event severity: corrected" are the constant
# background RxErr/advisory traffic on this box and are counted, not failed.
lab=$1; since=${2:-"10 min ago"}
h=$(curl -s -m 5 -o /dev/null -w "%{http_code}" localhost:30141/health)
echo "GUARD $lab $(date -Is) health=$h"
for d in 80:03.1 82:00.0 83:01.0 83:02.0 84:00.0 81:00.0; do
  p=/sys/bus/pci/devices/0000:$d
  nf=$(grep -i TOTAL_ERR_NONFATAL $p/aer_dev_nonfatal 2>/dev/null | awk '{print $2}')
  f=$(grep -i TOTAL_ERR_FATAL $p/aer_dev_fatal 2>/dev/null | awk '{print $2}')
  echo "AER $d nonfatal=${nf:-na} fatal=${f:-na}"
done
log=$(timeout 15 journalctl -k --since "$since" --no-pager 2>/dev/null)
bad1=$(echo "$log" | grep -E "pciehp|Completion-Wait loop timed out" | grep -E "80:|82:|83:|84:|81:")
# GHES records: {id}[Hardware Error]: ... event severity: X ; flag ids with severity != corrected that mention our devices
bad2=$(echo "$log" | awk '
 match($0,/\{[0-9]+\}\[Hardware Error\]/){ id=substr($0,RSTART,RLENGTH);
   if ($0 ~ /event severity:/){ s=$0; sub(/.*event severity: */,"",s); sev[id]=s }
   if ($0 ~ /0000:8[0-4]:/) mine[id]=1; seen[id]=1 }
 END{ nc=0; c=0; for (i in seen){ if (i in mine){ if (sev[i]!="corrected"){print "NONCORR " i " sev=" sev[i]} else c++ } } print "corrected_records_ours=" c > "/dev/stderr" }' 2>/tmp/guard_c.$$)
echo "$(cat /tmp/guard_c.$$)"; rm -f /tmp/guard_c.$$
if [ -n "$bad1$bad2" ]; then echo "KERNEL_BAD:"; echo "$bad1"; echo "$bad2" | tail -20; else echo "KERNEL_OK"; fi
dst=$(ps -eo stat=,comm= | awk '$1 ~ /^D/ && $2 ~ /khugepaged|kcompactd/')
[ -n "$dst" ] && echo "DSTATE: $dst" || echo "DSTATE_OK"
[ "$h" = 200 ] && [ -z "$bad1$bad2" ] && [ -z "$dst" ] && echo GUARD_PASS || echo GUARD_FAIL
