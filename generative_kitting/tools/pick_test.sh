#!/bin/bash
# Live pick check against the running bridge (port 8600).
# usage: pick_test.sh X Y TOP_Z [GRASP_LIFT] [YAW_DEG]
#   home -> collision world (target carved) -> approach 0.25 m above -> contact descent
#   -> adaptive close -> straight lift. Prints grasp/hold results.
#   YAW_DEG: gripper turn about world +Z (0 = pads close along world Y).
B=http://localhost:8600/api; L=$(ls -t "$HOME"/.nvidia-omniverse/logs/Kit/Isaac-Sim*/*/kit_*.log | head -1)   # newest Kit log
X=$1; Y=$2; Z=$3; GL=${4:-0.0}; AZ=$(python3 -c "print($Z + 0.25)")
YAW=$(python3 -c "import math; print(math.radians(${5:-0.0}))")
timeout 40 bash -c "until curl -s -m 2 $B/ping >/dev/null; do sleep 1; done"
curl -s -m 130 -X POST $B/home -d '{}' | python3 -c "import json,sys; print('home:', json.load(sys.stdin).get('status'))"
curl -s -m 180 -X POST $B/build_collision_world -d "{\"exclude_points\": [[$X, $Y, $Z]]}" | python3 -c "import json,sys; d=json.load(sys.stdin); print('world:', d.get('status') or d.get('error'), 'carved', d.get('carved'), 'inspect', d.get('inspect_sdf'))"
curl -s -m 180 -X POST $B/approach -d "{\"x\": $X, \"y\": $Y, \"z\": $AZ, \"yaw\": $YAW}" | python3 -c "import json,sys; d=json.load(sys.stdin); print('approach:', d.get('status') or d.get('error'))"
curl -s -m 180 -X POST $B/pick_descend -d "{\"x\": $X, \"y\": $Y, \"z\": $Z, \"grasp_lift\": $GL, \"yaw\": $YAW}" | python3 -c "import json,sys; d=json.load(sys.stdin); d.pop('wrist_image',None); print('descend:', d.get('status') or d.get('error'), 'contact', [s.get('contact') for s in d.get('steps',[]) if s.get('phase')=='grasp_descend'], 'xy drift mm', round(d.get('ee_drift_mm',{}).get('xy_total',-1),1), 'yaw off deg', [round(s.get('yaw_off_deg',0),1) for s in d.get('steps',[]) if s.get('phase')=='xy_align'])"
n0=$(wc -l < "$L")
CLOSE=$(curl -s -m 130 -X POST $B/pick_close -d '{}')
echo "$CLOSE" | python3 -c "import json,sys; d=json.load(sys.stdin); print('close: confirmed', d.get('grasp_confirmed'), 'target', round(d.get('finger_close',0),3), 'actual', d.get('actual_finger'), 'gap mm', round(1000*d.get('pad_gap_m',0),1), 'force N', round(d.get('contact_force',0),2))"
GH=$(echo "$CLOSE" | python3 -c "import json,sys; print(json.load(sys.stdin).get('grip_hold', 0.5))")
tail -n +$n0 "$L" | grep -a "ADAPTIVE" | cut -c60-240
curl -s -m 200 -X POST $B/pick_retract -d "{\"x\": $X, \"y\": $Y, \"z\": $Z, \"grip_hold\": $GH}" | python3 -c "import json,sys; d=json.load(sys.stdin); print('retract:', d.get('status') or d.get('error'), d.get('steps'))"
