#!/bin/sh
set -eu
action=$1
fault=$2
peer=$3
value=$4
chain=PHASE4_CHAOS
# Resolve the interface from the route; Compose interface numbering is not stable.
dev=$(ip route get "$peer" | awk '{for(i=1;i<=NF;i++) if($i=="dev") {print $(i+1); exit}}')
case "$fault" in
  network-delay)
    if [ "$action" = clear ]; then
      if tc qdisc show dev "$dev" | grep -q 'qdisc prio 4a00:'; then
        tc qdisc del dev "$dev" root handle 4a00:
      fi
      exit 0
    fi
    if tc qdisc show dev "$dev" | grep -q 'qdisc prio 4a00:'; then
      echo 'Delay already active; clear it first.' >&2; exit 1
    fi
    # add, never replace: refuse to overwrite another root qdisc.
    tc qdisc add dev "$dev" root handle 4a00: prio bands 3 priomap 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    trap 'tc qdisc del dev "$dev" root handle 4a00: || true' EXIT
    tc qdisc add dev "$dev" parent 4a00:3 handle 4a30: netem delay "${value}ms"
    tc filter add dev "$dev" protocol ip parent 4a00: prio 1 u32 match ip dst "$peer/32" match ip protocol 6 0xff match ip dport 9090 0xffff flowid 4a00:3
    trap - EXIT
    ;;
  connection-drop)
    if [ "$action" = clear ]; then
      if iptables -w -S "$chain" >/dev/null 2>&1; then
        if iptables -w -C OUTPUT -j "$chain" 2>/dev/null; then
          iptables -w -D OUTPUT -j "$chain"
        fi
        iptables -w -F "$chain"
        iptables -w -X "$chain"
      fi
      exit 0
    fi
    iptables -w -N "$chain"
    trap 'iptables -w -F "$chain"; iptables -w -X "$chain"' EXIT
    # Reject every Nth opening SYN, starting with the first; no random sampling.
    iptables -w -A "$chain" -d "$peer" -p tcp --dport 9090 --syn -m statistic --mode nth --every "$value" --packet 0 -j REJECT --reject-with tcp-reset
    iptables -w -A OUTPUT -j "$chain"
    trap - EXIT
    ;;
  *) exit 2 ;;
esac
